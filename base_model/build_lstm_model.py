import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib
from base_miner.get_tao_price import get_data, scale_data
import pandas as pd

def create_base_lstm(X_scaler:MinMaxScaler, y_scaler:MinMaxScaler, X_scaled:np.ndarray, y_scaled:np.ndarray) -> float:
    model_name = "models/base_lstm"

    # Reshape input for LSTM
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=6))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    model.save(f'{model_name}.h5')

    # Predict the prices - this is just for a local test, this prediction just allows
    # miners to assess the performance of their models on real data.
    predicted_prices = model.predict(X_test)

    # Rescale back to original range
    predicted_prices = y_scaler.inverse_transform(predicted_prices)
    y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(-1, 6))

    # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')
    
    return mse

def create_base_regression(scaler:MinMaxScaler, X_scaled:np.ndarray, y_scaled:np.ndarray) -> float:
    model_name = "models/base_linear_regression"

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = LinearRegression()
    model.fit(X_train, y_train)

    '''with h5py.File(f'{model_name}.h5', 'w') as hf:
        hf.create_dataset('coefficients', data=model.coef_)
        hf.create_dataset('intercept', data=model.intercept_)'''
    joblib.dump(model, f"{model_name}.joblib")

    # Predict the prices - this is just for a local test, this prediction just allows
    # miners to assess the performance of their models on real data.
    predicted_prices = model.predict(X_test)

    # Rescale back to original range
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')
    
    return mse

if __name__ == '__main__':
    input, output = get_data()
    
    output = output.dropna()
    input = input[input['Datetime'].isin(output['Datetime'])]
    
    input = input.dropna()
    output = output[output['Datetime'].isin(input['Datetime'])]
    
    X_scaler, y_scaler, X_data, y_data = scale_data(input, output)
    
    input['Datetime'] = pd.to_datetime(input['Datetime'])

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    # data.to_csv('mining_models/base_miner_data.csv')
    input = input[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']]
    type = 'lstm'

    if(type != 'regression'):
        input = np.array(input, dtype = np.float32).reshape(1, -1)
        input = np.reshape(input, (-1, 1, input.shape[1]))
        print(input)

    if(type == 'lstm'): create_base_lstm(X_scaler = X_scaler, y_scaler = y_scaler, X_scaled = X_data, y_scaled = y_data)