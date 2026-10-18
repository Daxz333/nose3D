import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

from head_reconstruction import api


class DenseHeadApiTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(api.head_api)
        self.client = app.test_client()

    def test_unvalidated_environment_cannot_fall_back_to_coarse(self):
        with patch.object(api, 'dense_readiness', return_value=False), patch.object(api, 'readiness', return_value=True):
            response = self.client.post('/api/head/dense/reconstruct', data={'image': (io.BytesIO(b'photo'), 'photo.jpg')})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json['success'])

    def test_invalid_print_dimensions_are_rejected(self):
        with patch.object(api, 'dense_readiness', return_value=True):
            for height in ('149', '251', 'nan', 'inf', 'unknown'):
                response = self.client.post('/api/head/dense/reconstruct', data={'height_mm': height})
                self.assertEqual(response.status_code, 400, height)

    def test_stl_is_served_as_download_with_expected_bytes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(api, 'OUTPUT', Path(folder)):
            name = 'a' * 32 + '.stl'
            (Path(folder) / name).write_bytes(b'known-stl')
            response = self.client.get('/api/head/models/' + name + '?download=1')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b'known-stl')
            self.assertIn('attachment', response.headers['Content-Disposition'])
            response.close()


if __name__ == '__main__':
    unittest.main()
