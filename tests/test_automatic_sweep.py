import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evals.automatic_sweep import gate, THRESHOLDS, rejection_reasons
from evals.channel_estimation import smooth


class GateTests(unittest.TestCase):
    def fixture(self, root, passes=True):
        (root/'report').mkdir();(root/'channels').mkdir();(root/'groupings').mkdir()
        def js(path,data):path.write_text(json.dumps(data))
        js(root/'progress.json',dict(status='complete'))
        js(root/'config.json',dict(training_clips=64,validation_clips=16,
            groups=[2],partition_seeds=[7],policies=[['c25_r0',25,0]]))
        js(root/'report'/'summary.json',dict(final=True,training_clips=64,validation_clips=16,thresholds=THRESHOLDS))
        counts=np.array([[900,100],[100,900]])
        channel=smooth(counts,.0001)
        mapping=np.arange(8192)%2
        np.save(root/'groupings'/'m2_seed7.npy',mapping)
        label='m2_seed7_c25_r0'
        np.savez(root/'channels'/f'{label}.npz',groups=mapping,channel=channel,counts=counts,
                 lower_95=channel-.01,upper_95=channel+.01,row_clips=np.array([64,64]))
        js(root/'channels'/f'{label}.json',dict(training_clips=64,validation_clips=16,
            partition_sha256=hashlib.sha256(mapping.tobytes()).hexdigest()))
        row=dict(groups=2,partition_seed=7,policy='c25_r0',min_row_pairs=1000 if passes else 10,
            min_row_clips=64,max_cell_ci_halfwidth=.01,full_vs_half_tv=.005,independent_halves_tv=.01,
            train_validation_tv=.01,final_vs_half_loss_delta_low=-.002,
            final_vs_half_loss_delta_high=.002,gain_ci_low=.2,training_pairs=2000)
        with (root/'report'/'diagnostics.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)

    def test_gate_freezes_passing_matrix_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            self.fixture(source)
            out=Path(tmp)/'sweep'
            plan=gate(source,out)
            self.assertEqual(plan['status'],'approved')
            self.assertEqual(len(plan['accepted']),1)
            self.assertTrue((out/'gate.txt').exists())
            archive=out/plan['accepted'][0]['archive']
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                gate(source,out)

    def test_gate_blocks_failed_collection_or_bad_quality(self):
        for failed in [False,True]:
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/'source';source.mkdir()
                self.fixture(source,passes=False)
                if failed:(source/'progress.json').write_text('{"status":"failed"}')
                plan=gate(source,Path(tmp)/'sweep')
                self.assertEqual(plan['status'],'blocked')
                self.assertFalse(plan['accepted'])

    def test_missing_diagnostics_or_threshold_relaxation_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();self.fixture(source)
            (source/'report'/'diagnostics.csv').write_text('groups,partition_seed,policy\n')
            self.assertEqual(gate(source,Path(tmp)/'sweep')['status'],'blocked')
            summary=json.loads((source/'report'/'summary.json').read_text())
            summary['thresholds']['min_row_pairs']=1
            (source/'report'/'summary.json').write_text(json.dumps(summary))
            self.assertEqual(gate(source,Path(tmp)/'other')['status'],'blocked')

    def test_nonfinite_quality_is_rejected(self):
        self.assertTrue(rejection_reasons(dict(min_row_pairs=float('nan'))))


class SamplerTests(unittest.TestCase):
    def test_direct_group_cdf_and_position_only_replay(self):
        import torch
        from watermark.position_grouped import PositionGroupedSampler,replay_groups,position_uniform
        mapping=torch.tensor([1,0,1,0])
        p=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
        sampler=PositionGroupedSampler({i:i for i in range(4)},b'key',mapping,top_k=0,top_p=1.)
        with patch('watermark.position_grouped.position_uniform',return_value=.1):
            output=sampler(torch.tensor([[9]]),p.log()[None]).argmax().item()
            self.assertEqual(int(mapping[output]),0)
            self.assertEqual(replay_groups(sampler.trace(),b'key').tolist(),[0])
        sampler=PositionGroupedSampler({i:i for i in range(4)},b'key',mapping,top_k=0,top_p=1.)
        for history in [[9],[9,1],[9,1,0]]:
            sampler(torch.tensor([history]),p.log()[None])
        trace=sampler.trace()
        self.assertNotIn('sample_id',trace)
        self.assertEqual(replay_groups(trace,b'key').tolist(),trace['selected_groups'].tolist())
        self.assertEqual(position_uniform(b'key',5),position_uniform(b'key',5))
        self.assertNotEqual(position_uniform(b'key',5),position_uniform(b'key',6))

    def test_ordinary_control_never_consults_prf(self):
        import torch
        from watermark.position_grouped import PositionGroupedSampler
        sampler=PositionGroupedSampler({0:0,1:1},b'independent-detector-key',[0,1],
                                        record_only=True,top_k=0,top_p=1.)
        with patch('watermark.position_grouped.position_uniform',side_effect=AssertionError('PRF consulted')):
            for _ in range(20):
                sampler(torch.tensor([[9]]),torch.tensor([[0.,0.]]))
        self.assertEqual(sampler.trace()['generation_mode'],'ordinary')

    def test_stateful_scoring_uses_absolute_positions_and_batch_arrival(self):
        import torch
        from watermark.position_grouped import PositionGroupedSampler
        from evals.automatic_sweep import score_observation
        sampler=PositionGroupedSampler({0:0,1:1},b'key',[0,1],top_k=0,top_p=1.)
        for _ in range(4):sampler(torch.tensor([[9]]),torch.tensor([[0.,0.]]))
        trace=sampler.trace()
        positions=np.array([0,2,3])
        observation=dict(positions=positions,tokens=trace['original_tokens'].numpy()[positions],
                         batches=np.array([0,1,1]),arrivals=np.array([.5,1.,1.]),
                         records=[dict(arrival=.5,seconds=.1),dict(arrival=1.,seconds=.1)])
        result,path=score_observation(trace,observation,b'key',np.array([[.9,.1],[.1,.9]]),.4)
        self.assertEqual(result['detection_seconds'],1.)
        self.assertEqual(result['missing_tokens'],1)
        np.testing.assert_array_equal(path[:,0],positions)
        np.testing.assert_allclose(path[:,1],np.arange(1,4)*np.log(1.8))


if __name__=='__main__':unittest.main()
