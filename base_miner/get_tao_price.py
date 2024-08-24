import yfinance as yf
import ta
from pandas import DataFrame
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_data() -> DataFrame:
    data = yf.download('TAO22974-USD', period = '1mo', interval = '5m')
    
    data['SMA_50'] = data['Close'].rolling(window = 50).mean()
    data['SMA_200'] = data['Close'].rolling(window = 200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['Momentum'] = ta.momentum.ROCIndicator(data['Close']).roc()
    for i in range(1, 7):
        data[f'NextClose{i}'] = data['Close'].shift(-1 * i)
    
    data.dropna(inplace = True)
    data.reset_index(inplace = True)
    
    return data

def scale_data(data: DataFrame) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']].values

    # Prepare target variable
    y = data[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].values

    y = y.reshape(-1, 6)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return scaler, X_scaled, y_scaled