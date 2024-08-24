import datetime
from base_miner.get_tao_price import get_data, scale_data
import pandas as pd
from datetime import timedelta
import numpy as np

def predict(timestamp: datetime, model, type) -> float:
    data = get_data()
    scaler, _, _ = scale_data(data)
    
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    current_time = datetime.fromisoformat(timestamp)
    interval_minutes = 5
    pred_time = current_time - timedelta(minutes=current_time.minute % interval_minutes,
                                seconds=current_time.second,
                                microseconds=current_time.microsecond)

    matching_row = data[data['Datetime'] == pred_time]

    print(pred_time, matching_row)

    # Check if matching_row is empty
    if matching_row.empty:
        print("No matching row found for the given timestamp.")
        return 0.0

    # data.to_csv('mining_models/base_miner_data.csv')
    input = matching_row[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']]

    if(type != 'regression'):
        input = np.array(input, dtype = np.float32).reshape(1, -1)
        input = np.reshape(input, (1, 1, input.shape[1]))
        print(input)

    prediction = model.predict(input)
    if(type!='regression'):
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))

    print(prediction)
    return prediction