import yfinance as yf
import ta
from pandas import DataFrame

def get_data() -> DataFrame:
    data = yf.download('TAO22974-USD', period = '1mo', interval = '5m')
    
    data['SMA_50'] = data['Close'].rolling(window = 50).mean()
    data['SMA_200'] = data['Close'].rolling(window = 200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['Momentum'] = ta.momentum.ROCIndicator(data['Close']).roc()
    for i in range(1, 7):
        data[f'NextClose{i}'] = data['Close'].shift(-1 * i)