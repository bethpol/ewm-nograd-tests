"""Collect ordinary Spark-TTS audio and validate group channels before watermark runs."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from evals.chunked_grouped_core import HOP, SR, encode_windows, policies
from watermark.grouped import random_groups

GROUPS = (2, 4, 8, 16)
PARTITIONS = (20261104, 20261105, 20261106)
POLICIES = policies()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def collect(args):
    import soundfile as sf
    import torch
    from evals.keyed_sparktts import sparktts_classes
    from evals.sparktts_transition import generate_nonempty

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [p.strip() for p in Path(args.prompt_file).read_text().splitlines() if p.strip()]
    if len(set(prompts)) != len(prompts) or len(prompts) < 8:
        raise ValueError('Need at least eight distinct prompt lines')
    # Split by text, rather than letting repetitions of a text cross the split.
    permutation = np.random.default_rng(args.seed).permutation(len(prompts))
    nval = max(2, len(prompts)//4)
    split_prompts = dict(validation=[prompts[i] for i in permutation[:nval]],
                         training=[prompts[i] for i in permutation[nval:]])
    references = [p for p in sorted(Path(args.reference_dir).glob('*.wav'))
                  if p.with_suffix('.txt').exists()]
    if not references:
        raise ValueError('No reference recordings with transcripts')
    config = dict(version=1, training_clips=args.training_clips,
                  validation_clips=args.validation_clips, seed=args.seed,
                  model_dir=str(Path(args.model_dir).resolve()), steps=args.steps,
                  temperature=1., top_k=50, top_p=.95, sample_rate=SR, hop_samples=HOP,
                  policies=POLICIES, left_context_tokens=100, groups=GROUPS,
                  partition_seeds=PARTITIONS, beta=args.beta, bootstrap=args.bootstrap,
                  split_prompts=split_prompts,
                  references=[dict(path=str(p.resolve()), transcript=p.with_suffix('.txt').read_text().strip())
                              for p in references],
                  sampling='ordinary model sampling; no watermark or keyed sampler',
                  alignment='known start; window origins on 320-sample grid; no inferred shifts',
                  preprocessing='existing codec wrapper; per-window volume normalization',
                  scope='pooled group channels under this prompt/voice distribution; shared voices, disjoint texts')
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Configuration differs from existing run; choose a new directory')
    write_json(config_path, config)
    maps_dir = out/'groupings'
    maps_dir.mkdir(exist_ok=True)
    for seed in PARTITIONS:
        for m in GROUPS:
            np.save(maps_dir/f'm{m}_seed{seed}.npy', random_groups(8192, m, seed))
    torch.set_num_threads(8)
    SparkTTS, Wrapper = sparktts_classes()
    device = torch.device(args.device)
    model = SparkTTS(Path(args.model_dir), device=device)
    encoder = Wrapper(args.model_dir, device, audio_tokenizer=model.audio_tokenizer)
    encoder.tok.model.eval()
    encoder.tok.feature_extractor.eval()

    def encode(audio):
        with torch.inference_mode():
            return encoder.encode_semantic(torch.from_numpy(audio.copy())).cpu().numpy()

    completed = 0
    total = args.training_clips + args.validation_clips
    started = time.monotonic()
    try:
        # Validation first makes fixed held-out diagnostics available at every
        # subsequent training checkpoint. Validation never supplies fitted counts.
        for split, target in [('validation', args.validation_clips), ('training', args.training_clips)]:
            for index in range(target):
                folder = out/'samples'/split/f'clip_{index:04d}'
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/'complete.json').exists():
                    completed += 1
                    continue
                text = split_prompts[split][index % len(split_prompts[split])]
                reference = references[index % len(references)]
                seed = args.seed + (1000000 if split == 'validation' else 0) + index
                write_json(out/'progress.json', dict(status='collecting', split=split,
                    index=index, completed=completed, target=total,
                    elapsed_seconds=time.monotonic()-started))
                clip_started = time.monotonic()
                wav_path = folder/'generated.wav'
                if wav_path.exists() and (folder/'metadata.json').exists() and (folder/'original.npy').exists():
                    audio, sr = sf.read(wav_path, dtype='float32')
                    original = np.load(folder/'original.npy')
                    if sr != SR or audio.ndim != 1:
                        raise ValueError('Cached audio has invalid format')
                else:
                    def operation():
                        with torch.inference_mode():
                            return model.inference(text=text, prompt_speech_path=reference,
                                prompt_text=reference.with_suffix('.txt').read_text().strip(),
                                max_new_tokens=args.steps, temperature=1., top_k=50, top_p=.95,
                                wm_enable=False, return_semantic_ids=True)
                    def retry(event):
                        with (out/'retries.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(split=split, index=index, **event))+'\n')
                    (audio, original), actual_seed, attempts = generate_nonempty(operation, seed, retry)
                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    original = original.cpu().numpy().reshape(-1)
                    if len(audio) < 400 or not len(original):
                        raise ValueError('Empty or too-short generated audio')
                    sf.write(wav_path, audio, SR, subtype='FLOAT')
                    np.save(folder/'original.npy', original)
                    write_json(folder/'metadata.json', dict(text=text, reference=str(reference),
                        seed=actual_seed, retries=attempts, seconds=len(audio)/SR,
                        original_tokens=len(original), sample_rate=SR))
                for name, chunk, look in POLICIES:
                    target_path = folder/f'{name}.npz'
                    if target_path.exists():
                        continue
                    encoded = encode_windows(audio, encode, chunk, left=100, look=look)
                    records = encoded.pop('records')
                    np.savez_compressed(target_path, **encoded)
                    write_json(folder/f'{name}_timing.json', records)
                write_json(folder/'complete.json', dict(seconds=time.monotonic()-clip_started))
                completed += 1
                print(f'CLIP {split} {index+1}/{target}; {len(original)} tokens; '
                      f'{time.monotonic()-clip_started:.2f}s', flush=True)
                if split == 'training' and (index+1 in [32, 64, 128, 256, 512] or index+1 == target):
                    report(args, final=False)
        report(args, final=True)
        write_json(out/'progress.json', dict(status='complete', completed=completed, target=total,
                   elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        write_json(out/'progress.json', dict(status='failed', completed=completed,
                   target=total, error=repr(error)))
        raise


def smooth(counts, beta):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or beta <= 0:
        raise ValueError('Require square counts and positive beta')
    return (counts+beta)/(counts.sum(1, keepdims=True)+counts.shape[1]*beta)


def clip_counts(folder, mapping, name):
    original = np.load(folder/'original.npy')
    with np.load(folder/f'{name}.npz') as archive:
        positions, tokens = archive['positions'], archive['tokens']
    if len(positions) != len(tokens) or (len(positions) and (np.diff(positions) <= 0).any()):
        raise ValueError('Invalid absolute token positions')
    valid = (positions >= 0) & (positions < len(original))
    positions, tokens = positions[valid], tokens[valid]
    if ((tokens < 0) | (tokens >= len(mapping))).any():
        raise ValueError('Recovered token outside vocabulary')
    m = int(mapping.max())+1
    matrix = np.zeros((m,m), dtype=np.int64)
    np.add.at(matrix, (mapping[original[positions]], mapping[tokens]), 1)
    return matrix


def loss(counts, channel):
    return -(counts*np.log(channel)).sum()/counts.sum() if counts.sum() else float('nan')


def weighted_tv(c1, c2, weights):
    return float((weights * (.5*np.abs(c1-c2).sum(1))).sum())


def information(counts):
    joint = counts/counts.sum()
    product = joint.sum(1)[:,None]*joint.sum(0)[None,:]
    valid = joint > 0
    return float((joint[valid]*np.log2(joint[valid]/product[valid])).sum())


def bootstrap_matrices(per_clip, beta, bootstrap, seed=42):
    rng = np.random.default_rng(seed)
    n = len(per_clip)
    weights = rng.multinomial(n, np.full(n, 1/n), size=bootstrap)
    counts = (weights @ per_clip.reshape(n, -1)).reshape(bootstrap, *per_clip.shape[1:])
    channels = (counts+beta)/(counts.sum(2, keepdims=True)+counts.shape[2]*beta)
    return np.quantile(channels, [.025,.975], axis=0)


def report(args, final=True):
    out = Path(args.output_dir)
    config = json.loads((out/'config.json').read_text())
    train = sorted(p.parent for p in (out/'samples'/'training').glob('clip_*/complete.json'))
    validation = sorted(p.parent for p in (out/'samples'/'validation').glob('clip_*/complete.json'))
    if len(train) < 2 or not validation:
        raise ValueError('Need at least two training and one validation recording')
    root = out/'report'
    root.mkdir(exist_ok=True)
    channels_dir = out/'channels'
    channels_dir.mkdir(exist_ok=True)
    checkpoints = [n for n in [16,32,64,128,256,512] if n < len(train)] + [len(train)]
    learning, diagnostics, heatmaps = [], [], {}
    rng = np.random.default_rng(7741)
    # Identical bootstrap draws across configurations preserve paired comparisons.
    val_weights = rng.multinomial(len(validation), np.full(len(validation),1/len(validation)),
                                 size=args.bootstrap)
    for m in config['groups']:
        for seed in config['partition_seeds']:
            mapping = np.load(out/'groupings'/f'm{m}_seed{seed}.npy')
            for name, chunk, look in config['policies']:
                training = np.stack([clip_counts(p, mapping, name) for p in train])
                val = np.stack([clip_counts(p, mapping, name) for p in validation])
                vtotal = val.sum(0)
                previous = None
                for n in checkpoints:
                    counts = training[:n].sum(0)
                    channel = smooth(counts, args.beta)
                    weights = counts.sum(1)/counts.sum()
                    split = n//2
                    half_tv = weighted_tv(smooth(training[:split].sum(0),args.beta),
                                          smooth(training[split:n].sum(0),args.beta), weights)
                    learning.append(dict(groups=m, partition_seed=seed, policy=name,
                        training_clips=n, training_pairs=int(counts.sum()),
                        validation_pairs=int(vtotal.sum()), heldout_log_loss=loss(vtotal,channel),
                        independent_halves_tv=half_tv,
                        successive_tv=(weighted_tv(channel,previous,weights) if previous is not None else None)))
                    previous = channel
                counts = training.sum(0)
                channel = smooth(counts, args.beta)
                n = len(train)//2
                half_channel = smooth(training[:n].sum(0),args.beta)
                weights = counts.sum(1)/counts.sum()
                half_change = weighted_tv(channel,half_channel,weights)
                baseline = (counts.sum(0)+args.beta)/(counts.sum()+m*args.beta)
                clip_losses = -(val*np.log(channel)).sum((1,2))
                clip_baseline = -(val*np.log(baseline[None,:])).sum((1,2))
                sizes = val.sum((1,2))
                gain_boot = (val_weights @ (clip_baseline-clip_losses))/(val_weights @ sizes)
                gain_ci = np.quantile(gain_boot,[.025,.975])
                delta_clip = -(val*np.log(channel/half_channel)).sum((1,2))
                delta_boot = (val_weights @ delta_clip)/(val_weights @ sizes)
                delta_ci = np.quantile(delta_boot,[.025,.975])
                if final:
                    lo,hi = bootstrap_matrices(training,args.beta,args.bootstrap,seed)
                    ci_halfwidth = float(((hi-lo)/2).max())
                else:
                    lo,hi = np.full_like(channel,np.nan),np.full_like(channel,np.nan)
                    ci_halfwidth = None
                row_counts = counts.sum(1)
                row_clips = (training.sum(2)>0).sum(0)
                half_tv = weighted_tv(smooth(training[:n].sum(0),args.beta),
                                      smooth(training[n:].sum(0),args.beta), weights)
                # Predeclared screening thresholds, not a mathematical guarantee.
                stable = bool(final and row_counts.min()>=500 and row_clips.min()>=32
                              and ci_halfwidth<=.05 and half_change<=.02 and half_tv<=.05
                              and delta_ci[0]>=-.01 and delta_ci[1]<=.01)
                informative = bool(gain_ci[0]>0)
                train_val_tv = weighted_tv(channel,smooth(vtotal,args.beta),weights)
                stationary = bool(train_val_tv<=.10)
                result = dict(groups=m, partition_seed=seed, policy=name,
                    training_clips=len(train), validation_clips=len(validation),
                    training_pairs=int(counts.sum()),validation_pairs=int(vtotal.sum()),
                    min_row_pairs=int(row_counts.min()), min_row_clips=int(row_clips.min()),
                    max_cell_ci_halfwidth=ci_halfwidth, full_vs_half_tv=half_change,
                    independent_halves_tv=half_tv, train_validation_tv=train_val_tv,
                    heldout_log_loss=loss(vtotal,channel),
                    heldout_gain_over_marginal_nats=float((clip_baseline-clip_losses).sum()/sizes.sum()),
                    gain_ci_low=float(gain_ci[0]),gain_ci_high=float(gain_ci[1]),
                    final_vs_half_loss_delta_low=float(delta_ci[0]),
                    final_vs_half_loss_delta_high=float(delta_ci[1]),
                    validation_recovery=float(np.trace(vtotal)/vtotal.sum()),
                    validation_plugin_mi_bits=information(vtotal),
                    stable_candidate=stable, informative=informative,
                    train_validation_consistent=stationary,
                    candidate_for_later_review=stable and informative and stationary)
                diagnostics.append(result)
                identity = f'm{m}_seed{seed}_{name}'
                np.savez_compressed(channels_dir/f'{identity}.npz', groups=mapping,
                    counts=counts, validation_counts=vtotal, channel=channel,
                    lower_95=lo,upper_95=hi,row_clips=row_clips)
                write_json(channels_dir/f'{identity}.json',dict(**result,beta=args.beta,
                    orientation='row=original group; column=recovered group',
                    partition_sha256=hashlib.sha256(mapping.tobytes()).hexdigest(),
                    uncertainty='pointwise 95% percentile intervals from recording-cluster bootstrap; not simultaneous',
                    fit_split='training only',config=str((out/'config.json').resolve())))
                heatmaps[m,seed,name]=channel
    save_csv(root/'learning_curves.csv',learning)
    save_csv(root/'diagnostics.csv',diagnostics)
    ready = [dict(groups=r['groups'],seed=r['partition_seed'],policy=r['policy'])
             for r in diagnostics if r['candidate_for_later_review']]
    write_json(root/'summary.json',dict(final=final,training_clips=len(train),
        validation_clips=len(validation),candidate_count=len(ready),candidates=ready,
        watermark_jobs_submitted=False,
        note='Screening only. No automatic selection or watermark launch. Channel memory/context and group mixture remain approximations.',
        thresholds=dict(min_row_pairs=500,min_row_clips=32,max_cell_ci_halfwidth=.05,
            full_vs_half_tv=.02,independent_halves_tv=.05,
            final_vs_half_loss_delta_ci_within=[-.01,.01],train_validation_tv=.10)))
    plot_report(root,config,learning,diagnostics,heatmaps,final)
    print(f'REPORT {len(train)} training, {len(validation)} validation; '
          f'{len(ready)}/{len(diagnostics)} candidates for later review',flush=True)


def plot_report(root,config,learning,diagnostics,heatmaps,final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip([p[0] for p in config['policies']],plt.get_cmap('tab10').colors))
    common = (f"Data: {config['training_clips']} target training and {config['validation_clips']} target validation ordinary, unwatermarked Spark-TTS clips. "
        f"Actual completed counts appear in summary.json and CSV files. Prompt texts are disjoint between splits; reference voices are shared. "
        "Known recording start; paired by nominal absolute token index, without alignment search. Missing window frames are unscored and never compacted. "
        "Different policies can therefore score different positions. Each policy uses its own training counts. "
        "cC_rR means C-token chunks and R-token right lookahead, at 50 nominal tokens/sec; all chunk policies use 100 tokens of left context. "
        "full means whole-recording encoding. Standard per-window codec preprocessing is retained. "
        f"Groups are fixed random vocabulary-count-balanced partitions with seeds {config['partition_seeds']}; they are not probability-balanced or learned. "
        f"Rows are original groups and columns recovered groups. Smoothing: (N_ab+{config['beta']})/(N_a+M*{config['beta']}). "
        "Channels are pooled stationary approximations, whose within-group token mixture may differ during watermark generation. "
        "This is channel estimation, not watermark detection, a false-positive test, or a proof of sufficient estimation.\n")
    def save(fig,stem,explanation):
        fig.tight_layout()
        for extension in ['png','pdf']:
            fig.savefig(root/f'{stem}.{extension}',dpi=160,bbox_inches='tight')
        (root/f'{stem}.txt').write_text(explanation+'\n\n'+common)
        plt.close(fig)

    fig,axes=plt.subplots(4,2,figsize=(13,15),squeeze=False)
    for row,m in enumerate(config['groups']):
        for name,_,_ in config['policies']:
            subset=[r for r in learning if r['groups']==m and r['policy']==name]
            ns=sorted(set(r['training_clips'] for r in subset))
            for col,metric in enumerate(['heldout_log_loss','independent_halves_tv']):
                values=np.array([[r[metric] for r in subset if r['training_clips']==n] for n in ns])
                ax=axes[row,col]
                ax.plot(ns,values.mean(1),label=name,color=colors[name],marker='o',markersize=3)
                ax.fill_between(ns,values.min(1),values.max(1),color=colors[name],alpha=.12)
                ax.set(xlabel='Training recordings',ylabel=('Held-out log loss (nats/pair)' if col==0 else 'Weighted row total variation'),
                       title=f'{m} groups: '+('predictive fit' if col==0 else 'independent-half stability'))
                ax.set_xscale('log',base=2)
        axes[row,0].legend(fontsize=7,ncol=2)
        axes[row,1].axhline(.05,color='black',ls='--',lw=1)
    save(fig,'channel_convergence',
        'Training-size convergence. Source: learning_curves.csv. Rows show 2, 4, 8, and 16 groups. '
        'The x-axis is the number of training recordings (base-2 log scale). Left: token-pair-weighted negative log probability '
        'on the same fixed validation recordings; lower is better. Right: row-wise TV distance between channels fitted to '
        'disjoint first and second halves of the training prefix, averaged using full-prefix source-group frequencies; lower is more stable. '
        'Each colored line is a policy, averaged over three partitions. Shading is their min-to-max range, NOT a confidence interval. '
        'The dashed TV=.05 line is a predeclared screening tolerance. Learning curves are descriptive and are not necessarily monotone.')

    if final:
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for m in config['groups']:
            rows=[r for r in diagnostics if r['groups']==m]
            for ax,metric,label in zip(axes,['min_row_pairs','max_cell_ci_halfwidth','full_vs_half_tv'],
                    ['Minimum training pairs in any source row','Largest cell interval half-width','Final vs half-data weighted row TV']):
                names=[p[0] for p in config['policies']]
                values=np.array([[r[metric] for r in rows if r['policy']==name] for name in names])
                ax.plot(range(len(names)),values.mean(1),label=f'{m} groups',marker='o')
                ax.fill_between(range(len(names)),values.min(1),values.max(1),alpha=.12)
                ax.set(xticks=range(len(names)),xticklabels=names,ylabel=label)
                ax.tick_params(axis='x',rotation=45)
                ax.legend()
        axes[0].axhline(500,color='black',ls='--'); axes[0].set_yscale('log')
        axes[1].axhline(.05,color='black',ls='--'); axes[2].axhline(.02,color='black',ls='--')
        save(fig,'channel_support_uncertainty',
            'Training support and uncertainty. Source: diagnostics.csv. X-axis: encoding policy. Curves: group counts; '
            'lines average three partitions and shaded ranges show partition min/max, not confidence bands. '
            'Left: smallest source-group training count; the log y-axis and dashed 500-pair line indicate row support. '
            'Center: maximum across matrix cells of half the pointwise 95% percentile bootstrap interval width. '
            f'The bootstrap resamples whole training recordings {config["bootstrap"]} times to retain within-recording dependence; '
            'these intervals are not simultaneous. Dashed .05 is the screening tolerance. '
            'Right: final channel versus first-half training channel, weighted row TV; dashed .02 is the screening tolerance. '
            'More support, narrower uncertainty and lower TV favor a stable estimate. Thresholds are practical screens, not formal guarantees.')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    names=[p[0] for p in config['policies']]
    for m in config['groups']:
        rows=[r for r in diagnostics if r['groups']==m]
        for ax,metric in zip(axes,['heldout_gain_over_marginal_nats','validation_plugin_mi_bits']):
            values=np.array([[r[metric] for r in rows if r['policy']==name] for name in names])
            ax.plot(range(len(names)),values.mean(1),label=f'{m} groups',marker='o')
            ax.fill_between(range(len(names)),values.min(1),values.max(1),alpha=.12)
            ax.set(xticks=range(len(names)),xticklabels=names)
            ax.tick_params(axis='x',rotation=40)
            ax.legend()
    axes[0].set_ylabel('Held-out gain over output marginal (nats/pair)')
    axes[0].axhline(0,color='black',ls='--')
    axes[1].set_ylabel('Validation plug-in mutual information (bits/pair)')
    save(fig,'channel_predictive_information',
        'Channel predictive usefulness. Source: diagnostics.csv. X-axis: encoding policy; lines average partitions and '
        'shading is partition min/max, not confidence intervals. Left: held-out log-loss improvement over a training-fitted '
        'output marginal that ignores the source group. Positive values mean the source group improves out-of-sample prediction. '
        'Right: empirical mutual information computed directly from validation transition counts, in bits per aligned pair. '
        'This plug-in MI can be upward biased, especially for larger group counts, and is not the achieved watermark log-growth rate. '
        'Recording-bootstrap intervals for predictive gain are in diagnostics.csv.')

    for m in config['groups']:
        fig,axes=plt.subplots(3,len(names),figsize=(25,10),squeeze=False)
        for row,seed in enumerate(config['partition_seeds']):
            for col,name in enumerate(names):
                ax=axes[row,col]
                channel=heatmaps[m,seed,name]
                im=ax.imshow(channel,vmin=0,vmax=1,cmap='Blues')
                if m<=4:
                    for a in range(m):
                        for b in range(m):
                            ax.text(b,a,f'{channel[a,b]:.2f}',ha='center',va='center',fontsize=8,
                                    color='white' if channel[a,b]>.6 else 'black')
                ax.set(title=f'{name}\npartition {seed}',xlabel='Recovered group',ylabel='Original group')
                ticks = range(m) if m <= 8 else np.linspace(0, m-1, 4, dtype=int)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
        # Fixed color scale is common to every panel; no shared-colorbar layout warnings.
        fig.subplots_adjust(right=.94)
        bar=fig.add_axes([.955,.15,.01,.7]);fig.colorbar(im,cax=bar,label='Conditional probability')
        # tight_layout does not support this explicit colorbar axis.
        for extension in ['png','pdf']:
            fig.savefig(root/f'channels_{m}_groups.{extension}',dpi=160,bbox_inches='tight')
        (root/f'channels_{m}_groups.txt').write_text(
            f'Training-fitted {m}-group transition matrices. Rows of panels are the three fixed partitions; '
            'columns are encoding policies. Within each panel the y-axis is the original group and x-axis the recovered group. '
            'Color is smoothed conditional probability on a shared 0-to-1 scale. Matrix rows sum to one. '
            'Diagonal mass indicates same-group recovery; off-diagonal structure describes group substitutions. '
            'Values are annotated for two and four groups; exact counts, probabilities, and final bootstrap intervals are in ../channels/. '
            'No validation observations are used to fit these matrices.\n\n'+common)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['collect','report'])
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--model_dir',default='checkpoints/sparktts/Spark-TTS-0.5B')
    parser.add_argument('--prompt_file',default='data/sparktts_channel_prompts.txt')
    parser.add_argument('--reference_dir',default='data/audio_prompts')
    parser.add_argument('--training_clips',type=int,default=512)
    parser.add_argument('--validation_clips',type=int,default=128)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--seed',type=int,default=202611040)
    parser.add_argument('--beta',type=float,default=1e-4)
    parser.add_argument('--bootstrap',type=int,default=300)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.training_clips<2 or args.validation_clips<1 or args.bootstrap<20 or args.beta<=0:
        parser.error('Need >=2 training clips, >=1 validation clip, >=20 bootstrap replicates, beta>0')
    if args.command=='report':
        config=json.loads((Path(args.output_dir)/'config.json').read_text())
        args.beta=config['beta'];args.bootstrap=config['bootstrap']
        report(args)
    else:
        collect(args)

if __name__=='__main__':
    main()
