"""Geometry checks for the dense, closed print head; real inference is a separate smoke run."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from head_reconstruction.dense_geometry import ROOT, template_data, apply_dense_prediction, export_head


class DenseHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = template_data()

    def test_real_template_is_one_closed_volume(self):
        mesh = trimesh.Trimesh(self.template['vertices'], self.template['faces'], process=False)
        self.assertTrue(mesh.is_volume)
        self.assertEqual(mesh.body_count, 1)
        self.assertGreater(len(mesh.vertices), 100_000)

    def test_exterior_vertices_stay_bitwise_identical(self):
        template = self.template
        mean = template['mean_face']
        prediction = dict(mid_vertices=mean + [.01, 0, .02], mean_shape=mean,
                          displacement=np.zeros((1, 256, 256), dtype=np.float32),
                          uv=np.zeros((len(mean), 2)))
        changed = apply_dense_prediction(prediction, template)
        frozen = template['weights'] == 0
        self.assertGreater(np.count_nonzero(frozen), 1000)
        np.testing.assert_array_equal(changed[frozen], template['vertices'][frozen])
        self.assertGreater(np.max(abs(changed - template['vertices'])), .01)

    def test_learned_displacement_changes_actual_vertices(self):
        template = self.template
        mean = template['mean_face']
        prediction = dict(mid_vertices=mean.copy(), mean_shape=mean,
                          displacement=np.full((1, 256, 256), .01, dtype=np.float32),
                          uv=np.full((len(mean), 2), .5))
        changed = apply_dense_prediction(prediction, template)
        self.assertGreater(np.max(np.linalg.norm(changed - template['vertices'], axis=1)), .005)

    def test_stl_height_and_topology_survive_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            for height in (150, 250):
                output = Path(folder) / str(height)
                metadata = export_head(output, height_mm=height)
                mesh = trimesh.load(output.with_suffix('.stl'))
                self.assertTrue(mesh.is_volume)
                self.assertEqual(mesh.body_count, 1)
                self.assertAlmostEqual(mesh.extents[2], height, places=4)
                self.assertAlmostEqual(mesh.bounds[0, 2], 0, places=4)
                self.assertEqual(metadata['topology']['boundary_edges'], 0)
                self.assertEqual(metadata['topology']['nonmanifold_edges'], 0)
                self.assertEqual(metadata['topology']['self_intersection_triangles'], 0)

    def test_print_range_is_enforced(self):
        with self.assertRaises(ValueError):
            export_head('unused.glb', height_mm=300)

    @unittest.skipUnless((ROOT / 'outputs/head_dense/sample_prediction.npz').is_file(), 'Requires the local HRN smoke prediction')
    def test_real_prediction_exports_repaired_closed_print_mesh(self):
        prediction = dict(np.load(ROOT / 'outputs/head_dense/sample_prediction.npz'))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'real-face'
            metadata = export_head(output, prediction=prediction, height_mm=150)
            mesh = trimesh.load(output.with_suffix('.stl'))
            self.assertTrue(mesh.is_volume)
            self.assertEqual(mesh.body_count, 1)
            self.assertEqual(metadata['topology']['self_intersection_triangles'], 0)
            self.assertLess(metadata['topology']['fixed_exterior_drift_mm'], .005)
            self.assertAlmostEqual(mesh.extents[2], 150, places=4)


if __name__ == '__main__':
    unittest.main()
