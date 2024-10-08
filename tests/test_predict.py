import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from base_miner.get_tao_price import get_data, scale_data
from base_miner.predict import predict

# Mock data for get_data function
mock_data = pd.DataFrame({
    'Open': [100, 101, 102],
    'High': [101, 102, 103],
    'Low': [99, 100, 101],
    'Close': [100, 101, 102],
    'Volume': [1000, 1100, 1200],
    'SMA_50': [100, 100, 100],
    'SMA_200': [101, 101, 101],
    'RSI': [50, 50, 50],
    'CCI': [100, 100, 100],
    'Momentum': [1, 2, 3],
    'Datetime': pd.to_datetime(['2024-10-03 12:00:00', '2024-10-03 12:05:00', '2024-10-03 12:10:00'])
})

def test_predict():
    # Mock the get_data and scale_data functions
    with patch('base_miner.predict.get_data', return_value=(mock_data, None)) as mock_get_data:
        with patch('base_miner.predict.scale_data', return_value=(None, None, None, None)) as mock_scale_data:
            # Mock the model
            model = MagicMock()
            model.predict.return_value = np.array([1])

            # Call the predict function
            prediction = predict('2024-10-03T12:10:00', model, 'regression')

            # Check if the model's predict function was called
            model.predict.assert_called_once()

            # Check if the prediction is correct
            assert prediction == np.array([1])

            # Check if the get_data function was called
            mock_get_data.assert_called_once()

            # Check if the scale_data function was called
            mock_scale_data.assert_called_once()