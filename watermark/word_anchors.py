"""Causal word-commitment policies and synchronization diagnostics, without hashing."""

from collections import deque
from difflib import SequenceMatcher
import math
import unicodedata


def activation_token(observation, delay_samples, hop):
    """Never schedule inside tokens already consumed by a lookahead decoder."""
    available = observation["available_samples"]
    frontier = observation.get("generation_frontier_samples", available)
    if frontier < available:
        raise ValueError("Generation frontier cannot precede the emitted audio")
    return (frontier + delay_samples + hop - 1) // hop


def time_aligned_matches(current, previous, tolerance):
    """Monotone one-to-one exact-word matches within a timestamp tolerance.

    LCS permits earlier insertions/deletions without changing later word IDs.
    Timestamp gating distinguishes repeated occurrences of the same word.
    Both hypotheses must already be available to the caller.
    """
    n, m = len(current), len(previous)

    def compatible(i, j):
        a, b = current[i], previous[j]
        return (a["normalized"] == b["normalized"]
                and abs(a["start_samples"] - b["start_samples"]) <= tolerance
                and abs(a["end_samples"] - b["end_samples"]) <= tolerance)

    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1],
                                lengths[i - 1][j - 1] + int(compatible(i - 1, j - 1)))
    matches = set()
    i, j = n, m
    while i and j:
        if compatible(i - 1, j - 1) and lengths[i][j] == lengths[i - 1][j - 1] + 1:
            matches.add(i - 1)
            i, j = i - 1, j - 1
        elif lengths[i - 1][j] >= lengths[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matches


def normalize_word(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return "".join(c for c in text if c.isalnum() or c == "'").strip("'")


class WordCommitter:
    """Commit an agreeing prefix, withholding each hypothesis's trailing word.

    Every update must expose strictly more audio. Commits are irreversible; a
    later contradictory hypothesis is recorded and blocks further commits until
    the original committed prefix agrees again. No final-transcript repair.
    """

    def __init__(self, agreement=2, margin_samples=3200, delay_samples=1600, hop=320):
        if agreement < 2 or min(margin_samples, delay_samples) < 0 or hop < 1:
            raise ValueError("Invalid commitment parameters")
        self.history = deque(maxlen=agreement)
        self.agreement = agreement
        self.margin = margin_samples
        self.delay = delay_samples
        self.hop = hop
        self.last_samples = -1
        self.events = []
        self.contradictions = []

    def update(self, observation):
        available = observation["available_samples"]
        if available <= self.last_samples:
            raise ValueError("Updates must expose strictly more audio")
        self.last_samples = available
        words = [dict(w, normalized=normalize_word(w["word"])) for w in observation["words"]]
        words = [w for w in words if w["normalized"]]
        labels = [w["normalized"] for w in words]
        committed = [e["word"] for e in self.events]
        if labels[:len(committed)] != committed:
            self.contradictions.append({"available_samples": available,
                "committed": committed, "hypothesis": labels})
            self.history.clear()
            return []
        # Withhold the last word even if the recognizer gives it an end time.
        eligible = words[:-1]
        self.history.append([w["normalized"] for w in eligible])
        if len(self.history) < self.agreement:
            return []
        common = 0
        for values in zip(*self.history):
            if len(set(values)) != 1:
                break
            common += 1
        new = []
        for index in range(len(self.events), common):
            word = eligible[index]
            if word["end_samples"] > available - self.margin:
                break
            event = {"word": word["normalized"], "word_index": index,
                     "word_start_samples": word["start_samples"],
                     "word_end_samples": word["end_samples"],
                     "confirmed_samples": available,
                     "generation_frontier_samples": observation.get("generation_frontier_samples", available),
                     "activation_token": activation_token(observation, self.delay, self.hop)}
            self.events.append(event)
            new.append(event)
        return new


class LocalWordCommitter(WordCommitter):
    """Confirm new word occurrences without requiring an identical old prefix.

    Agreement is checked against the last N eligible hypotheses, independently
    for each candidate. Once emitted, events never change. A midpoint frontier
    and interval-overlap check prevent re-emitting old occurrences, while
    tolerating small overlaps in ASR estimates of adjacent word boundaries.
    No counterpart stream, final audit, or future observation is consulted.
    """

    def __init__(self, *args, timestamp_tolerance_samples=3200, **kwargs):
        super().__init__(*args, **kwargs)
        if timestamp_tolerance_samples < 0:
            raise ValueError("Timestamp tolerance must be nonnegative")
        self.timestamp_tolerance = timestamp_tolerance_samples

    def update(self, observation):
        available = observation["available_samples"]
        if available <= self.last_samples:
            raise ValueError("Updates must expose strictly more audio")
        self.last_samples = available
        words = [dict(w, normalized=normalize_word(w["word"])) for w in observation["words"]]
        words = [w for w in words if w["normalized"]]
        committed = [e["word"] for e in self.events]
        labels = [w["normalized"] for w in words]
        if labels[:len(committed)] != committed:
            self.contradictions.append({"available_samples": available,
                "committed": committed, "hypothesis": labels,
                "action": "record_only; old prefix does not block new anchors"})
        eligible = words[:-1]
        self.history.append(eligible)
        if len(self.history) < self.agreement:
            return []
        stable = set(range(len(eligible)))
        for previous in list(self.history)[:-1]:
            stable &= time_aligned_matches(eligible, previous, self.timestamp_tolerance)
        frontier = ((self.events[-1]["word_start_samples"] + self.events[-1]["word_end_samples"]) / 2
                    if self.events else -1)
        new = []
        for index, word in enumerate(eligible):
            start, end = word["start_samples"], word["end_samples"]
            if (index not in stable or (start + end) / 2 <= frontier or end <= start
                    or end > available - self.margin):
                continue
            # A revision of an already emitted occurrence must not emit it again.
            # Use temporal overlap regardless of spelling; half the shorter
            # interval distinguishes revisions from a slightly overlapping neighbor.
            if any(max(0, min(end, e["word_end_samples"]) - max(start, e["word_start_samples"]))
                   >= .5 * min(end - start, e["word_end_samples"] - e["word_start_samples"])
                   for e in self.events):
                continue
            event = {"word": word["normalized"], "word_index": len(self.events),
                     "word_start_samples": start, "word_end_samples": end,
                     "confirmed_samples": available,
                     "generation_frontier_samples": observation.get("generation_frontier_samples", available),
                     "activation_token": activation_token(observation, self.delay, self.hop)}
            self.events.append(event)
            new.append(event)
            frontier = (start + end) / 2
        return new


def context_timeline(events, token_count, context_words=1):
    """Absolute token indices; simultaneous commits expose the latest context."""
    if context_words < 1:
        raise ValueError("context_words must be positive")
    timeline = [()] * token_count
    words = []
    for event in events:
        words.append(event["word"])
        start = event["activation_token"]
        context = tuple(words[-context_words:])
        for t in range(start, token_count):
            timeline[t] = context
    return timeline


def agreement_statistics(embedder_events, detector_events, token_count, context_words):
    a = context_timeline(embedder_events, token_count, context_words)
    b = context_timeline(detector_events, token_count, context_words)
    active = sum(bool(x or y) for x, y in zip(a, b))
    both = sum(bool(x and y) for x, y in zip(a, b))
    matches = sum(bool(x or y) and x == y for x, y in zip(a, b))
    both_matches = sum(bool(x and y) and x == y for x, y in zip(a, b))
    # Offline matching is for measurement only; it never changes live events.
    matcher = SequenceMatcher(a=[e["word"] for e in embedder_events],
                              b=[e["word"] for e in detector_events], autojunk=False)
    differences = []
    for block in matcher.get_matching_blocks():
        for j in range(block.size):
            x, y = embedder_events[block.a + j], detector_events[block.b + j]
            differences.append(abs(x["activation_token"] - y["activation_token"]))
    return {"token_positions": token_count, "active_union_positions": active,
            "both_active_positions": both, "matching_active_positions": matches,
            "active_context_agreement": matches / active if active else None,
            "both_active_context_agreement": both_matches / both if both else None,
            "embedder_active_fraction": sum(bool(x) for x in a) / token_count,
            "detector_active_fraction": sum(bool(x) for x in b) / token_count,
            "embedder_committed_words": len(embedder_events),
            "detector_committed_words": len(detector_events),
            "matched_committed_words": len(differences),
            "activation_index_absolute_differences": differences}
