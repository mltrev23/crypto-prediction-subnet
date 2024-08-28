import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_and_save_base_model_lstm(scaler:MinMaxScaler, X_scaled:np.ndarray, y_scaled:np.ndarray) -> float:
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
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 6))

    # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')
    
    return mse