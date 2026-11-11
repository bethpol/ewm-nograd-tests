"""Fail-closed channel-quality gate and position-only grouped watermark sweep."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from evals.channel_estimation import write_json, save_csv
from evals.chunked_grouped_core import Evidence, encode_windows, policies, SR

THRESHOLDS = dict(min_row_pairs=500, min_row_clips=32, max_cell_ci_halfwidth=.05,
                  full_vs_half_tv=.02, independent_halves_tv=.05,
                  final_vs_half_loss_delta_ci_within=[-.01,.01], train_validation_tv=.10)


def rejection_reasons(row):
    reasons=[]
    for name,minimum in [('min_row_pairs',500),('min_row_clips',32)]:
        if not np.isfinite(row.get(name,float('nan'))) or row[name]<minimum:
            reasons.append(f'{name} below {minimum}')
    for name,maximum in [('max_cell_ci_halfwidth',.05),('full_vs_half_tv',.02),
                         ('independent_halves_tv',.05),('train_validation_tv',.10)]:
        if not np.isfinite(row.get(name,float('nan'))) or row[name]>maximum:
            reasons.append(f'{name} above {maximum} or unavailable')
    lo=row.get('final_vs_half_loss_delta_low',float('nan'))
    hi=row.get('final_vs_half_loss_delta_high',float('nan'))
    if not np.isfinite([lo,hi]).all() or lo<-.01 or hi>.01 or lo>hi:
        reasons.append('final-vs-half held-out loss interval not within [-.01,.01]')
    gain=row.get('gain_ci_low',float('nan'))
    if not np.isfinite(gain) or gain<=0:
        reasons.append('held-out predictive gain lower interval is not positive')
    return reasons


def gate(source, destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    plan_path=destination/'gate.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if plan['source']!=str(source):
            raise ValueError('Cannot resume with a different source')
        if plan['status']=='approved':
            # Frozen channels must never silently change on resume.
            for item in plan['accepted']:
                path=destination/item['archive']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Frozen channel changed after gate approval')
        return plan
    plan=dict(source=str(source),thresholds=THRESHOLDS,accepted=[],rejected=[],status='blocked')
    try:
        progress=json.loads((source/'progress.json').read_text())
        config=json.loads((source/'config.json').read_text())
        summary=json.loads((source/'report'/'summary.json').read_text())
        if progress['status']!='complete' or not summary['final']:
            raise ValueError('Channel collection did not complete successfully with a final report')
        if (summary['training_clips']!=config['training_clips']
                or summary['validation_clips']!=config['validation_clips']):
            raise ValueError('Completed split sizes do not match collection targets')
        if summary['thresholds']!=THRESHOLDS:
            raise ValueError('Channel report screening thresholds differ from the declared gate')
        with (source/'report'/'diagnostics.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        expected={(m,s,p[0]) for m in config['groups'] for s in config['partition_seeds']
                  for p in config['policies']}
        seen=set()
        frozen=destination/'frozen_channels'
        frozen.mkdir(exist_ok=True)
        for raw in rows:
            row={}
            for k,v in raw.items():
                if v in ['True','False']:
                    row[k]=v=='True'
                elif k=='policy':
                    row[k]=v
                else:
                    row[k]=float(v) if v else float('nan')
            identity=(int(row['groups']),int(row['partition_seed']),row['policy'])
            if identity in seen or identity not in expected:
                raise ValueError('Duplicate or unexpected channel configuration')
            seen.add(identity)
            m,seed,name=identity
            label=f'm{m}_seed{seed}_{name}'
            reasons=rejection_reasons(row)
            if not reasons:
                archive=source/'channels'/f'{label}.npz'
                metadata=json.loads(archive.with_suffix('.json').read_text())
                with np.load(archive) as data:
                    mapping=data['groups']; channel=data['channel'];counts=data['counts']
                    low,high=data['lower_95'],data['upper_95']
                    if (channel.shape!=(m,m) or counts.shape!=(m,m) or (counts<0).any()
                            or mapping.shape!=(8192,) or set(mapping.tolist())!=set(range(m))
                            or not np.isfinite(channel).all() or (channel<=0).any()
                            or not np.allclose(channel.sum(1),1)):
                        raise ValueError(f'Invalid channel archive: {label}')
                    exact_mapping=np.load(source/'groupings'/f'm{m}_seed{seed}.npy')
                    if not np.array_equal(mapping,exact_mapping):
                        raise ValueError(f'Partition mismatch: {label}')
                    if (int(counts.sum())!=int(row['training_pairs'])
                            or int(counts.sum(1).min())!=int(row['min_row_pairs'])
                            or int(data['row_clips'].min())!=int(row['min_row_clips'])
                            or low.shape!=channel.shape or high.shape!=channel.shape
                            or not np.isfinite(low).all() or not np.isfinite(high).all()
                            or (low>high).any()
                            or not np.isclose(((high-low)/2).max(),row['max_cell_ci_halfwidth'])):
                        raise ValueError(f'Archive and diagnostics disagree: {label}')
                if (metadata['training_clips']!=config['training_clips']
                        or metadata['validation_clips']!=config['validation_clips']
                        or metadata['partition_sha256']!=hashlib.sha256(mapping.tobytes()).hexdigest()):
                    raise ValueError(f'Incomplete or mismatched channel metadata: {label}')
                target=frozen/archive.name
                shutil.copyfile(archive,target)
                shutil.copyfile(archive.with_suffix('.json'),target.with_suffix('.json'))
                plan['accepted'].append(dict(groups=m,partition_seed=seed,policy=name,
                    archive=str(target.relative_to(destination)),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            else:
                plan['rejected'].append(dict(groups=m,partition_seed=seed,policy=name,reasons=reasons))
        if seen!=expected:
            raise ValueError('Final diagnostics are missing channel configurations')
        plan['status']='approved' if plan['accepted'] else 'blocked'
        plan['reason']='passing channels frozen' if plan['accepted'] else 'no channels pass all quality screens'
        write_json(destination/'channel_collection_config.json',config)
    except (OSError,ValueError,KeyError,TypeError) as error:
        plan['accepted']=[]
        plan['reason']=str(error)
    write_json(plan_path,plan)
    (destination/'gate.txt').write_text(
        f"Automatic gate status: {plan['status']}. Reason: {plan['reason']}.\n"
        f"Accepted channels: {len(plan['accepted'])}; rejected channels: {len(plan['rejected'])}.\n"
        'Only complete channel collections with final reports are eligible. Every accepted channel must pass all '
        'predeclared support, uncertainty, training stability, held-out predictive gain, and train/validation consistency checks. '
        'Criteria are in gate.json; per-channel failures are listed there. Accepted matrices and partitions are copied '
        'to frozen_channels with SHA256 hashes before any watermark evaluation. No criteria are relaxed automatically. '
        'If none pass, no model is loaded and no watermark audio is generated. These are practical screening criteria, '
        'not a proof of channel accuracy or anytime-validity.\n')
    return plan


def load_key(path):
    import secrets
    import os
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(secrets.token_bytes(32))
    return path.read_bytes()


def score_observation(trace, observation, key, channel, alpha):
    from watermark.position_grouped import replay_groups, masses
    expected=replay_groups(trace,key).numpy()
    mapping=trace['groups'].numpy()
    valid=(observation['positions']>=0)&(observation['positions']<len(expected))
    positions=observation['positions'][valid]
    tokens=observation['tokens'][valid]
    batches=observation['batches'][valid]
    arrivals=observation['arrivals'][valid]
    if len(tokens) and ((tokens<0).any() or (tokens>=len(mapping)).any()):
        raise ValueError('Recovered ID outside vocabulary')
    detector=Evidence(channel,alpha)
    # Group masses are reconstructed from saved generation probabilities, not
    # from an autoregressive model fed noisy recovered history.
    for batch in np.unique(batches):
        use=batches==batch
        ps=positions[use]
        probability=np.stack([masses(trace['p0'][int(t)],trace['groups']).numpy() for t in ps])
        detector.update(ps,mapping[tokens[use]],expected[ps],probability,float(arrivals[use][-1]))
    record_end=observation['records'][-1]['arrival']
    path=np.asarray(detector.path,dtype=float).reshape(-1,3)
    return dict(detected=detector.first_detection is not None,
        detection_seconds=detector.first_detection,final_log_e=detector.wealth,
        scored_tokens=len(positions),missing_tokens=len(expected)-len(positions),
        extra_tokens=int((~valid).sum()),duration_seconds=float(record_end),
        token_group_recovery=(float(np.mean(mapping[tokens]==mapping[trace['original_tokens'].numpy()[positions]]))
                              if len(positions) else None),
        encoding_seconds=sum(r['seconds'] for r in observation['records'])),path


def run(args,plan):
    import soundfile as sf
    import torch
    from evals.keyed_sparktts import sparktts_classes
    from evals.sparktts_transition import generate_nonempty
    from watermark.position_grouped import PositionGroupedSampler,replay_groups
    out=Path(args.output_dir).resolve()
    config=json.loads((out/'channel_collection_config.json').read_text())
    settings=dict(clips_per_group_partition=args.clips,ordinary_controls_per_group_partition=args.null_clips,
        steps=args.steps,alpha=args.alpha,generation_seed=args.seed,model_dir=config['model_dir'],
        prompt_file=str(Path(args.prompt_file).resolve()),
        prompts=[p.strip() for p in Path(args.prompt_file).read_text().splitlines() if p.strip()],
        seed_rule='HMAC-SHA256(recording_key, absolute_semantic_token_index_uint64_be)',
        key_policy='independent private key per recording; index is the only PRF message',
        detector='saved-generation-distribution replay; fixed nominal index alignment',
        attacks='none; clean round trip only',
        source_channels=plan['source'],quality_gate=getattr(args, 'quality_gate_description',
            'gate.json; frozen before evaluation'))
    if not settings['prompts']:
        raise ValueError('No evaluation prompts')
    settings_path=out/'sweep_config.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Sweep resume settings differ')
    write_json(settings_path,settings)
    pairs=sorted(set((r['groups'],r['partition_seed']) for r in plan['accepted']))
    targets={pair:[r for r in plan['accepted'] if (r['groups'],r['partition_seed'])==pair] for pair in pairs}
    total=len(pairs)*(args.clips+args.null_clips)
    write_json(out/'progress.json',dict(status='loading_model',target_recordings=total,completed_recordings=0))
    torch.set_num_threads(8)
    SparkTTS,Wrapper=sparktts_classes()
    device=torch.device('cuda')
    model=SparkTTS(Path(config['model_dir']),device=device)
    encoder=Wrapper(config['model_dir'],device,audio_tokenizer=model.audio_tokenizer)
    encoder.tok.model.eval();encoder.tok.feature_extractor.eval()
    def encode(wave):
        with torch.inference_mode():
            return encoder.encode_semantic(torch.from_numpy(wave.copy())).cpu().numpy()
    completed=0
    for m,seed in pairs:
        pair=f'm{m}_seed{seed}'
        channels={}
        for item in targets[m,seed]:
            with np.load(out/item['archive']) as archive:
                channels[item['policy']]=archive['channel'].copy()
                mapping=archive['groups'].copy()
        for mode,number in [('watermarked',args.clips),('ordinary',args.null_clips)]:
            for index in range(number):
                folder=out/'samples'/pair/mode/f'clip_{index:04d}'
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').exists():
                    completed+=1
                    continue
                write_json(out/'progress.json',dict(status='running',pair=pair,mode=mode,index=index,
                    completed_recordings=completed,target_recordings=total))
                key_path=Path('.cache')/'automatic_sweep_keys'/out.name/pair/mode/f'clip_{index:04d}.key'
                if (folder/'trace.pt').exists() and not key_path.exists():
                    raise ValueError('Private recording key missing; cannot resume reproducibly')
                key=load_key(key_path)
                wrong=hashlib.sha256(b'wrong-key-control\0'+key).digest()
                text=settings['prompts'][index%len(settings['prompts'])]
                reference=config['references'][index%len(config['references'])]
                ordinary_seed=args.seed+seed%10000*10000+m*1000+index+(5000000 if mode=='ordinary' else 0)
                if (folder/'trace.pt').exists() and (folder/'generated.wav').exists():
                    trace=torch.load(folder/'trace.pt',map_location='cpu',weights_only=True)
                    audio,sr=sf.read(folder/'generated.wav',dtype='float32')
                    if sr!=SR or audio.ndim!=1:
                        raise ValueError('Invalid cached audio')
                else:
                    def operation():
                        sampler=PositionGroupedSampler(model.semantic_to_vocab,key,mapping,
                            record_only=mode=='ordinary',temperature=1.,top_k=50,top_p=.95)
                        with torch.inference_mode():
                            wave,original=model.inference(text=text,
                                prompt_speech_path=Path(reference['path']),prompt_text=reference['transcript'],
                                max_new_tokens=args.steps,wm_enable=False,
                                return_semantic_ids=True,keyed_sampler=sampler)
                        trace=sampler.trace()
                        if not torch.equal(original.cpu(),trace['original_tokens']):
                            raise ValueError('Trace differs from generated semantic tokens')
                        if mode=='watermarked' and not torch.equal(replay_groups(trace,key),trace['groups'][original.cpu()]):
                            raise ValueError('Direct group-CDF replay failed')
                        return wave,trace
                    def retry(event):
                        with (out/'retries.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(pair=pair,mode=mode,index=index,**event))+'\n')
                    (audio,trace),actual_seed,retries=generate_nonempty(operation,ordinary_seed,retry)
                    audio=np.asarray(audio,dtype=np.float32).reshape(-1)
                    torch.save(trace,folder/'trace.pt')
                    sf.write(folder/'generated.wav',audio,SR,subtype='FLOAT')
                    write_json(folder/'metadata.json',dict(text=text,reference=reference['path'],
                        key_file=str(key_path.resolve()),generation_seed=actual_seed,retries=retries,
                        mode=mode,original_tokens=len(trace['original_tokens']),seconds=len(audio)/SR))
                if mode=='watermarked' and not torch.equal(replay_groups(trace,key), trace['selected_groups']):
                    raise ValueError('Cached trace/key replay mismatch')
                results=[]
                for name,chunk,look in config['policies']:
                    if name not in channels:
                        continue
                    observation_path=folder/f'{name}_observation.npz'
                    timing_path=folder/f'{name}_timing.json'
                    if observation_path.exists() and timing_path.exists():
                        with np.load(observation_path) as archive:
                            observation={k:archive[k].copy() for k in archive.files}
                        observation['records']=json.loads(timing_path.read_text())
                    else:
                        observation=encode_windows(audio,encode,chunk,left=100,look=look)
                        np.savez_compressed(observation_path,**{k:v for k,v in observation.items() if k!='records'})
                        write_json(timing_path,observation['records'])
                    estimators=[('matched',channels[name])]
                    if name!='full' and 'full' in channels:
                        estimators.append(('full_channel_mismatch',channels['full']))
                    for fit,channel in estimators:
                        for key_name,detect_key in [('same_key',key),('wrong_key',wrong)]:
                            metrics,path=score_observation(trace,observation,detect_key,channel,args.alpha)
                            tag=f'{name}_{fit}_{key_name}'
                            np.save(folder/f'{tag}_path.npy',path)
                            results.append(dict(groups=m,partition_seed=seed,mode=mode,index=index,
                                policy=name,fit=fit,key=key_name,path=str((folder/f'{tag}_path.npy').relative_to(out)),**metrics))
                write_json(folder/'results.json',results)
                write_json(folder/'complete.json',dict(conditions=len(results)))
                completed+=1
                print(f'SWEEP {pair} {mode} {index+1}/{number}; {completed}/{total} recordings',flush=True)
        summarize(out)
    summarize(out)
    write_json(out/'progress.json',dict(status='complete',completed_recordings=completed,target_recordings=total))


def summarize(out):
    out=Path(out)
    rows=[]
    for marker in sorted((out/'samples').glob('*/*/clip_*/complete.json')):
        rows.extend(json.loads((marker.parent/'results.json').read_text()))
    if not rows:
        return
    root=out/'report';root.mkdir(exist_ok=True)
    save_csv(root/'per_clip.csv',rows)
    config=json.loads((out/'sweep_config.json').read_text())
    alpha=config['alpha']
    # One shared restricted horizon avoids treating short clip endings as evidence
    # of detection. All comparisons use the shortest duration in this report.
    horizon=min(r['duration_seconds'] for r in rows)
    grouped={}
    for row in rows:
        identity=tuple(row[k] for k in ['groups','policy','fit','mode','key'])
        grouped.setdefault(identity,[]).append(row)
    summary=[]
    for (m,policy,fit,mode,key),values in sorted(grouped.items()):
        times=np.array([r['detection_seconds'] if r['detected'] else np.inf for r in values])
        summary.append(dict(groups=m,policy=policy,fit=fit,mode=mode,key=key,clips=len(values),
            partitions=len(set(r['partition_seed'] for r in values)),horizon_seconds=horizon,
            detected_fraction_by_horizon=float(np.mean(times<=horizon)),
            restricted_mean_detection_seconds=float(np.minimum(times,horizon).mean()),
            mean_final_log_e=float(np.mean([r['final_log_e'] for r in values])),
            mean_missing_tokens=float(np.mean([r['missing_tokens'] for r in values]))))
    save_csv(root/'summary.csv',summary)
    write_json(root/'summary.json',dict(horizon_seconds=horizon,conditions=len(rows),
        completed_recordings=len(list((out/'samples').glob('*/*/clip_*/complete.json'))),
        note='Clean saved-waveform simulated arrival, saved p0 replay, screened configurations only. '
             'No numerical false-positive guarantee is inferred from finite controls.'))
    plot_sweep(out,root,rows,summary,horizon,alpha)


def plot_sweep(out,root,rows,summary,horizon,alpha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    groups=sorted(set(r['groups'] for r in rows))
    names=[p[0] for p in policies()]
    colors=dict(zip(names,plt.get_cmap('tab10').colors))
    common=(f'Alpha={alpha}; decision threshold log(1/alpha)={np.log(1/alpha):.4f}. '
        'Source: per_clip.csv, summary.csv, and saved per-recording path arrays. Only configurations admitted by gate.json appear. '
        'Group maps and channels were frozen before evaluation. Semantic tokens alone carry the watermark; global/control tokens are unchanged. '
        'PRF message is the absolute semantic-token index, continuous across chunks, with an independent private key per recording. '
        'No audio/STT/history seed extraction is used. Detection reconstructs group masses from saved generation distributions, '
        'so this is an oracle-distribution experiment, not a standalone blind detector. '
        'cC_rR denotes C-token audio chunks and R-token lookahead at 50 nominal tokens/sec, with 100 tokens of left context. '
        'full waits for the complete recording. Missing frames are skipped using absolute positions. '
        'Evidence is accumulated across chunks and tested only after each completed chunk. Arrival time includes right lookahead, '
        'but excludes measured encoder runtime and network delay. Audio was generated first and arrivals were simulated. '
        'Restricted comparisons use the shortest completed recording duration in the current report, '
        f'{horizon:.3f} seconds; undetected or later-detected recordings are assigned that horizon for restricted mean time. '
        'Different group/partition settings have fresh generations; encoding policies within a recording share its waveform. '
        'No attacks or alignment search are included. No confidence bands are shown; this bounded experiment does not establish false-positive control.\n')
    def save(fig,stem,description):
        fig.tight_layout()
        for extension in ['png','pdf']:
            fig.savefig(root/f'{stem}.{extension}',dpi=180,bbox_inches='tight')
        (root/f'{stem}.txt').write_text(description+'\n\n'+common)
        plt.close(fig)
    fig,axes=plt.subplots(len(groups),1,figsize=(10,4*len(groups)),squeeze=False)
    grid=np.linspace(0,horizon,150)
    for ax,m in zip(axes[:,0],groups):
        for name in names:
            selected=[r for r in rows if r['groups']==m and r['policy']==name and
                      r['fit']=='matched' and r['mode']=='watermarked' and r['key']=='same_key']
            if not selected:continue
            times=np.array([r['detection_seconds'] if r['detected'] else np.inf for r in selected])
            ax.step(grid,np.mean(times[:,None]<=grid[None,:],axis=0),where='post',
                    label=f'{name} (n={len(selected)})',color=colors[name])
        ax.set(xlabel='Audio received (seconds)',ylabel='Fraction detected',ylim=(-.02,1.02),title=f'{m} groups')
        ax.legend(fontsize=8,ncol=2)
    save(fig,'streaming_detection',
        'Cumulative detection fraction versus audio arrival time. Each panel is a group count and each curve an approved '
        'encoding policy, pooled with equal recording weight across available partitions. n is the number of recordings in that curve. '
        'Only watermarked, same-key, matched-channel conditions appear. Curves may include different partition subsets because '
        'quality screening is policy-specific; consult gate.json. A higher/earlier curve indicates faster detection.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for name in names:
        subset=[r for r in summary if r['policy']==name and r['fit']=='matched' and
                r['mode']=='watermarked' and r['key']=='same_key']
        if not subset:continue
        for ax,metric in zip(axes,['detected_fraction_by_horizon','restricted_mean_detection_seconds']):
            ax.plot([r['groups'] for r in subset],[r[metric] for r in subset],marker='o',label=name,color=colors[name])
            ax.set_xlabel('Number of token groups')
            ax.set_xticks(groups)
            ax.legend(fontsize=8,ncol=2)
    axes[0].set_ylabel('Detected fraction by shared horizon')
    axes[1].set_ylabel('Restricted mean detection time (seconds)')
    save(fig,'grouping_latency',
        'Grouping and latency comparison. Source: summary.csv. X-axis: number of token groups. '
        'Left: fraction crossing by the shared horizon; higher is better. Right: mean min(detection time, horizon), '
        'including failures; lower is better. Curves show encoding policies using matched channels on watermarked same-key audio. '
        'Missing points mean no screened configuration or no completed recordings. Partitions are pooled; screening can change their composition.')
    fig,axes=plt.subplots(len(groups),1,figsize=(11,4*len(groups)),squeeze=False)
    for ax,m in zip(axes[:,0],groups):
        for mode,key,style in [('watermarked','same_key','-'),('watermarked','wrong_key','--'),('ordinary','same_key',':')]:
            for name in names:
                selected=[r for r in rows if r['groups']==m and r['policy']==name and
                          r['fit']=='matched' and r['mode']==mode and r['key']==key]
                if not selected:continue
                times=np.array([r['detection_seconds'] if r['detected'] else np.inf for r in selected])
                ax.step(grid,np.mean(times[:,None]<=grid[None,:],axis=0),where='post',color=colors[name],ls=style)
        ax.set(xlabel='Audio received (seconds)',ylabel='Fraction crossing',title=f'{m} groups',ylim=(-.02,1.02))
        handles=[Line2D([0],[0],color=colors[n],label=n) for n in names if any(r['groups']==m and r['policy']==n for r in rows)]
        handles += [Line2D([0],[0],color='black',ls=s,label=l) for s,l in
                    [('-','Watermarked / same key'),('--','Watermarked / wrong key'),(':','Ordinary audio / independent key')]]
        ax.legend(handles=handles,fontsize=7,ncol=3)
    save(fig,'detection_controls',
        'Detection and controls on matched channels. Color identifies encoding policy. Solid lines: watermarked audio with '
        'the embedding key. Dashed: identical watermarked audio replayed with a different key. Dotted: genuine ordinary '
        'categorical generation, without consulting the detector key, then tested with an independent key. '
        'Y-axis is empirical fraction ever crossing by the indicated audio time. These small controls are diagnostic, '
        'not precise estimates at a 1% level and not a proof of sequential null assumptions.')
    fig,axes=plt.subplots(len(groups),1,figsize=(11,4*len(groups)),squeeze=False)
    for ax,m in zip(axes[:,0],groups):
        for name in names:
            if name=='full':continue
            for fit,style in [('matched','-'),('full_channel_mismatch','--')]:
                selected=[r for r in rows if r['groups']==m and r['policy']==name and r['fit']==fit and
                          r['mode']=='watermarked' and r['key']=='same_key']
                if not selected:continue
                times=np.array([r['detection_seconds'] if r['detected'] else np.inf for r in selected])
                ax.step(grid,np.mean(times[:,None]<=grid[None,:],axis=0),where='post',
                        color=colors[name],ls=style,label=f'{name} / {fit}')
        ax.set(xlabel='Audio received (seconds)',ylabel='Fraction detected',title=f'{m} groups',ylim=(-.02,1.02))
        if ax.lines:ax.legend(fontsize=7,ncol=2)
    save(fig,'channel_matching',
        'Channel-fit ablation on watermarked same-key audio. Each color is a chunk policy. Solid: channel fitted with '
        'the same chunk policy. Dashed: channel fitted with full-recording encoding. A dashed curve exists only where '
        'both that chunk channel and the full channel passed the gate for that partition. Subsets may differ. '
        'Both variants reuse exactly the same recovered observations within each eligible recording.')
    fig,axes=plt.subplots(len(groups),1,figsize=(11,4*len(groups)),squeeze=False)
    for ax,m in zip(axes[:,0],groups):
        primary=[r for r in rows if r['groups']==m and r['mode']=='watermarked' and r['key']=='same_key' and r['fit']=='matched']
        if primary:
            identity=min((r['partition_seed'],r['index']) for r in primary)
            for r in primary:
                if (r['partition_seed'],r['index'])!=identity:continue
                path=np.load(out/r['path'])
                if not len(path):continue
                # Display only final wealth at each arrival, matching the stopping rule.
                ends=np.r_[np.flatnonzero(np.diff(path[:,2])!=0),len(path)-1]
                ax.step(np.r_[0,path[ends,2]],np.r_[0,path[ends,1]],where='post',color=colors[r['policy']],label=r['policy'])
            ax.set_title(f'{m} groups; first completed partition/index {identity}')
        ax.axhline(np.log(1/alpha),color='black',ls='--',label='Detection threshold')
        ax.set(xlabel='Audio received (seconds)',ylabel='Cumulative log e-value')
        ax.legend(fontsize=8,ncol=2)
    save(fig,'example_evidence_paths',
        'Illustrative cumulative evidence paths. Each panel selects the lexicographically first completed watermarked '
        'partition/index for its group count, without selecting for detection success. Colors identify matched-channel '
        'encoding policies. Curves show wealth only at observation-arrival boundaries; black dashed is log(1/alpha). '
        'Evidence is not reset at chunks. Full trajectories continue after the first crossing for diagnostics; a deployed '
        'detector would stop then. These examples do not summarize population performance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--clips',type=int,default=16)
    parser.add_argument('--null_clips',type=int,default=16)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--alpha',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=202611060)
    parser.add_argument('--prompt_file',default='data/tts_prompts.txt')
    parser.add_argument('--gate_only',action='store_true')
    args=parser.parse_args()
    if args.clips<1 or args.null_clips<1 or args.steps<1 or not 0<args.alpha<1:
        parser.error('Need positive clip counts/steps and alpha in (0,1)')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    plan=gate(args.channel_dir,out)
    print(f"GATE {plan['status']}: {len(plan['accepted'])} channels; {plan['reason']}",flush=True)
    if plan['status']!='approved':
        (out/'status').write_text('blocked_by_channel_quality\n')
        return
    if args.gate_only:return
    try:
        (out/'status').write_text('running\n')
        run(args,plan)
        (out/'status').write_text('complete\n')
    except Exception as error:
        write_json(out/'error.json',dict(error=repr(error)))
        (out/'status').write_text('failed\n')
        raise

if __name__=='__main__':main()
