import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import patch

# Import the functions to be tested from your module
from base_miner.get_tao_price import get_data, scale_data

# Mock data for testing purposes
mock_data = {
    'Open': [100, 101, 102, 103, 104],
    'High': [101, 102, 103, 104, 105],
    'Low': [99, 100, 101, 102, 103],
    'Close': [100, 101, 102, 103, 104],
    'Volume': [1000, 1100, 1200, 1300, 1400]
}

def test_get_data():
    input_df, output_df = get_data()

    # Test if the input DataFrame contains expected columns
    expected_input_columns = ['SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']
    assert all(col in input_df.columns for col in expected_input_columns)

    # Test if the output DataFrame contains expected columns
    expected_output_columns = ['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']
    assert all(col in output_df.columns for col in expected_output_columns)

def test_scale_data():
    # Mock input data for the scaling function
    input_data = pd.DataFrame({
        'Open': [1, 2, 3, 4, 5],
        'High': [2, 3, 4, 5, 6],
        'Low': [1, 1.5, 2, 2.5, 3],
        'Volume': [1000, 1500, 2000, 2500, 3000],
        'SMA_50': [1, 1, 1, 1, 1],
        'SMA_200': [2, 2, 2, 2, 2],
        'RSI': [30, 40, 50, 60, 70],
        'CCI': [100, 150, 200, 250, 300],
        'Momentum': [1, 2, 3, 4, 5],
    })

    output_data = pd.DataFrame({
        'NextClose1': [1, 2, 3, 4, 5],
        'NextClose2': [2, 3, 4, 5, 6],
        'NextClose3': [3, 4, 5, 6, 7],
        'NextClose4': [4, 5, 6, 7, 8],
        'NextClose5': [5, 6, 7, 8, 9],
        'NextClose6': [6, 7, 8, 9, 10]
    })

    # Call the scale_data function
    X_scaler, y_scaler, X_scaled, y_scaled = scale_data(input_data, output_data)

    # Test if the scaling is done correctly
    assert isinstance(X_scaled, np.ndarray)
    assert isinstance(y_scaled, np.ndarray)

    # Check that the scaling transformed data properly within [0, 1] range
    assert X_scaled.min() >= 0 and X_scaled.max() <= 1
    assert y_scaled.min() >= 0 and y_scaled.max() <= 1

    # Test that the scaled data shapes are correct
    assert X_scaled.shape == (5, 9)  # 5 rows, 9 features
    assert y_scaled.shape == (5, 6)  # 5 rows, 6 targets
