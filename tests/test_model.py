import unittest
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "ann_weights.pkl")

class TestANNModel(unittest.TestCase):

    def test_scaler_file_exists(self):
        self.assertTrue(os.path.exists(SCALER_PATH), "Scaler file scaler.pkl missing")

    def test_weights_file_exists(self):
        self.assertTrue(os.path.exists(WEIGHTS_PATH), "ANN weights file ann_weights.pkl missing")

    def test_scaler_transform(self):
        scaler = joblib.load(SCALER_PATH)
        # Sample input matching feature columns (10 features)
        sample = np.array([[0.12, 240.5, 4.6, 0.0, 0.0, 1.0, 14, 15, 6, 2]])
        transformed = scaler.transform(sample)
        self.assertEqual(transformed.shape, (1, 10))

    def test_numpy_ann_forward_pass(self):
        from dashboard.app import NumpyANN
        weights = joblib.load(WEIGHTS_PATH)
        scaler = joblib.load(SCALER_PATH)
        ann = NumpyANN(weights)
        sample = np.array([[0.12, 240.5, 4.6, 0.0, 0.0, 1.0, 14, 15, 6, 2]])
        scaled_sample = scaler.transform(sample)
        pred = ann.predict(scaled_sample)
        self.assertEqual(pred.shape, (1, 1))
        self.assertGreater(float(pred[0][0]), 0.0)

if __name__ == "__main__":
    unittest.main()
