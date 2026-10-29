# References for a word-conditioned seed protocol

Recommended construction: HMAC-SHA-256 as a keyed pseudorandom function,
followed by discrete inverse-CDF sampling from the desired distribution. These
are two distinct operations. The word is public context, not the secret key.

## What to cite

Mihir Bellare, "New Proofs for NMAC and HMAC: Security without Collision
Resistance," Journal of Cryptology 28(4), 844--878, 2015,
[doi:10.1007/s00145-014-9185-x](https://doi.org/10.1007/s00145-014-9185-x).
An earlier version appears in CRYPTO 2006, pp. 602--619,
[doi:10.1007/11818175_36](https://doi.org/10.1007/11818175_36).
The author links the full version at [ePrint 2006/043](https://eprint.iacr.org/2006/043).
This is the relevant PRF-security reference: its guarantee assumes appropriate
pseudorandomness of the underlying compression function. It is not a proof
that concrete SHA-256 outputs are literally independent uniform random numbers.

For the original construction, cite Bellare, Canetti, and Krawczyk, "Keying Hash
Functions for Message Authentication," CRYPTO 1996, pp. 1--15,
[author PDF](https://cseweb.ucsd.edu/~mihir/papers/kmd5.pdf).

For the specific SHA-256 instantiation and implementation conformance tests,
use [RFC 4231](https://www.rfc-editor.org/rfc/rfc4231).

For the distribution-mapping step, cite Luc Devroye, *Non-Uniform Random Variate
Generation*, Springer-Verlag, 1986, Chapter III, Section 2, starting at p. 85.
The author provides the [discrete chapter PDF](https://luc.devroye.org/chapter_three.pdf)
and [book contents](https://luc.devroye.org/rnbookindex.html).

BibTeX entries are in `docs/word_seed_references.bib`.

## Proposed application to this watermark

For a fresh shared secret key K and a public recording identifier, use a
canonical length-prefixed encoding of:

```
protocol version | recording ID | stream ID | normalized word context |
absolute token index | draw role | optional expansion-block counter
```

Compute HMAC-SHA-256(K, encoding). The absolute token index prevents reusing a
draw throughout a word interval. Draw roles such as `group` and `within-group`
separate different uses of randomness. Hash neither an uncertain boundary
timestamp nor a reset-at-boundary position counter. The public recording ID
must be known to both sides; its recovery is a separate protocol decision.

Interpret pseudorandom bits as U in (0,1), then select the first group whose
cumulative probability exceeds U. For group probabilities g_t, this is
S_t = F^{-1}_{g_t}(U_t). Sampling a token conditionally inside the selected group
preserves the original token distribution under ideal uniform randomness.
The detector still needs the agreed g_t; hashing words does not reconstruct it.

For a simple floating-point implementation, use 52 hash bits as integer j and
U=(j+0.5)/2^52, then use the shared float64 CDF routine. This is finite-grid
sampling, not mathematically exact sampling from arbitrary real probabilities.
If exact finite-probability sampling is required, specify integer probability
weights and draw an unbiased integer with rejection using additional
domain-separated HMAC blocks, then select its cumulative-weight interval.
That version is exact for the represented rational probabilities under ideal
unbiased random bits; cryptographic pseudorandomness remains an assumption.

The papers justify the PRF and sampling components, not this complete
word-anchored audio watermark protocol or its anytime-validity theorem. State
the ideal-randomness/conditional-null assumptions separately and distinguish
cryptographic computational guarantees from exact e-process validity.
