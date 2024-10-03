import yfinance as yf
import ta
from pandas import DataFrame
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_data() -> DataFrame:
    input = yf.download('TAO22974-USD', period = '1mo', interval = '5m')
    
    input['SMA_50'] = input['Close'].rolling(window = 50).mean()
    input['SMA_200'] = input['Close'].rolling(window = 200).mean()
    input['RSI'] = ta.momentum.RSIIndicator(input['Close']).rsi()
    input['CCI'] = ta.trend.CCIIndicator(input['High'], input['Low'], input['Close']).cci()
    input['Momentum'] = ta.momentum.ROCIndicator(input['Close']).roc()
    output = DataFrame()
    for i in range(1, 7):
        output[f'NextClose{i}'] = input['Close'].shift(-1 * i)
    print(output)
    
    input.reset_index(inplace = True)
    
    return input, output

def scale_data(input: DataFrame, output: DataFrame) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    X = input[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']].values

    # Prepare target variable
    y = output[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].values

    y = y.reshape(-1, 6)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)

    return scaler, X_scaled, y_scaled