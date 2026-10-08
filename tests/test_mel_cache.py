import unittest

import torch

from utils.audio import hann_window, mel_basis, mel_spectrogram


class MelCacheTests(unittest.TestCase):
    def setUp(self):
        mel_basis.clear()
        hann_window.clear()
        self.audio = torch.linspace(-0.5, 0.5, 4096).unsqueeze(0)

    def tearDown(self):
        mel_basis.clear()
        hann_window.clear()

    def test_cached_results_match_fresh_results_across_audio_settings(self):
        configs = [
            (512, 40, 16000, 128, 512, 0, 8000),
            (1024, 80, 22050, 256, 1024, 0, 8000),
            (512, 40, 22050, 128, 512, 0, 8000),
            (512, 40, 16000, 128, 512, 100, 8000),
            (512, 40, 16000, 128, 256, 0, 8000),
        ]
        expected = []
        for config in configs:
            mel_basis.clear()
            hann_window.clear()
            expected.append(mel_spectrogram(self.audio, *config))
        mel_basis.clear()
        hann_window.clear()
        for config, reference in zip(configs, expected):
            with self.subTest(config=config):
                actual = mel_spectrogram(self.audio, *config)
                torch.testing.assert_close(actual, reference)
                torch.testing.assert_close(mel_spectrogram(self.audio, *config),
                                           reference)


if __name__ == "__main__":
    unittest.main()
