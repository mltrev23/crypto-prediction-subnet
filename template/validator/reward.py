# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
from typing import List
import bittensor as bt
from template.protocol import Challenge
import torch
import yfinance as yf
import time
from datetime import datetime, timedelta
from pytz import timezone

INTERVAL = 30

def reward(query: int, response: int) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    bt.logging.info(f"In rewards, query val: {query}, response val: {response}, rewards val: {1.0 if response == query * 2 else 0}")
    return 1.0 if response == query * 2 else 0


def get_rewards(
    self,
    query: Challenge,
    responses: List[Challenge],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    
    if len(responses) == 0:
        bt.logging.info("Got no responses. Returning reward tensor of zeros.")
        return [], torch.zeros_like(0).to(self.device)  # Fallback strategy: Log and return 0.
    data = yf.download('TAO22974-USD', period = '1mo', interval = '5m')
    
    timestamp = query.timestamp
    timestamp = datetime.fromisoformat(timestamp)
    
    # Round up current timestamp and then wait until that time has been hit
    rounded_up_time = timestamp - timedelta(minutes=timestamp.minute % INTERVAL,
                                    seconds=timestamp.second,
                                    microseconds=timestamp.microsecond) + timedelta(minutes=INTERVAL + 5, seconds=30)
    
    ny_timezone = timezone('America/New_York')
    
    while (datetime.now(ny_timezone) < rounded_up_time - timedelta(minutes=4, seconds=30)):
        bt.logging.info(f"Waiting for next {INTERVAL}m interval...")
        if(datetime.now(ny_timezone).minute%10==0):
            self.resync_metagraph()
        time.sleep(15)
        
    # current_time_adjusted = rounded_up_time - timedelta(minutes=INTERVAL + 5)
    # print(rounded_up_time, rounded_up_time.hour, rounded_up_time.minute, current_time_adjusted)
    # Prepare to extract close price for this timestamp
    
    data = yf.download('TAO22974-USD', period = '1d', interval = '5m')
    bt.logging.info("Procured data from yahoo finance.")
    
    
    
    # Get all the reward results by iteratively calling your reward() function.
    
    return np.array(
        [reward(query, response) for response in responses]
    )
