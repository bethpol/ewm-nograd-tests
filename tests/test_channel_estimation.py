import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evals.chunked_grouped_core import encode_windows, Evidence
from evals.channel_estimation import smooth, clip_counts, bootstrap_matrices, weighted_tv


class ChannelTests(unittest.TestCase):
    def test_boundary_loss_does_not_shift_next_chunk(self):
        # A mock of the real wav2vec2 convolutional length formula.
        def encode(wave):
            return np.arange(max(0, (len(wave)-400)//320+1))
        observation = encode_windows(np.zeros(320*100), encode, 25, left=0, look=0)
        np.testing.assert_array_equal(observation['positions'],
            np.concatenate([np.arange(a,a+24) for a in range(0,100,25)]))
        self.assertEqual(observation['tokens'][24], 0)
        self.assertEqual(observation['positions'][24], 25)
        self.assertEqual(observation['arrivals'][24], 1.)
        with_context = encode_windows(np.zeros(320*100), encode, 25, left=100, look=25)
        np.testing.assert_array_equal(with_context['positions'],np.arange(99))
        self.assertTrue((np.diff(with_context['arrivals']) >= 0).all())

    def test_unavailable_frames_are_not_compacted_in_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'original.npy',np.array([0,1,0,1,0]))
            np.savez(root/'test.npz',positions=np.array([0,2,4]),tokens=np.array([0,0,1]))
            result=clip_counts(root,np.array([0,1]),'test')
            np.testing.assert_array_equal(result,[[2,1],[0,0]])

    def test_channel_smoothing_and_recording_bootstrap(self):
        counts=np.array([[9,1],[0,0]])
        channel=smooth(counts,.1)
        np.testing.assert_allclose(channel.sum(1),1)
        np.testing.assert_allclose(channel[1],[.5,.5])
        per_clip=np.array([[[9,1],[1,9]],[[5,5],[5,5]],[[8,2],[2,8]]])
        lo,hi=bootstrap_matrices(per_clip,.1,100)
        self.assertEqual(lo.shape,(2,2))
        self.assertTrue((lo<=hi).all())
        self.assertTrue((lo>0).all())
        self.assertEqual(weighted_tv(channel,channel,np.array([.5,.5])),0)

    def test_evidence_accumulates_across_batches_but_checks_at_arrival(self):
        channel=np.array([[.9,.1],[.1,.9]])
        detector=Evidence(channel,alpha=.4)
        detector.update([0],[0],[0],[[.5,.5]],.5)
        self.assertIsNone(detector.first_detection)
        detector.update([2],[1],[1],[[.5,.5]],1.)
        self.assertAlmostEqual(detector.wealth,2*np.log(1.8))
        self.assertEqual(detector.first_detection,1.)
        with self.assertRaisesRegex(ValueError,'repeated evidence'):
            detector.update([2],[1],[1],[[.5,.5]],1.)
        neutral=Evidence(np.ones((1,1)))
        neutral.update([0,1],[0,0],[0,0],[[1],[1]],1.)
        self.assertEqual(neutral.wealth,0)

    def test_normalization_for_arbitrary_observed_group(self):
        channel=smooth(np.array([[8,2],[3,7]]),.01)
        mass=np.array([.2,.8])
        for y in range(2):
            factors=channel[:,y]/(mass@channel[:,y])
            self.assertAlmostEqual(mass@factors,1.)


if __name__=='__main__':
    unittest.main()
