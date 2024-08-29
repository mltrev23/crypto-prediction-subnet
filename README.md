# CryptoFlow Prediction Subnet

Welcome to the **CryptoFlow Prediction Subnet**! This project is focused on building an advanced prediction subnet within the Bittensor network, specializing in forecasting the prices of various cryptocurrencies such as TAO, BTC, ETH, and more.

## Overview

The CryptoFlow Prediction Subnet leverages state-of-the-art machine learning models and the decentralized Bittensor network to provide accurate and real-time predictions of cryptocurrency prices. By participating in this subnet, you contribute to a decentralized financial forecasting ecosystem that aims to provide valuable insights for traders, investors, and researchers.

### Importance of Crypto Price Prediction

Cryptocurrency markets are notoriously volatile, with prices fluctuating dramatically over short periods. Accurate price predictions are crucial for several reasons:

- **Informed Decision-Making**: Traders and investors rely on predictions to make informed decisions about buying, selling, or holding assets. Accurate predictions can significantly enhance profitability and reduce risk.
  
- **Market Efficiency**: By providing more accurate price forecasts, the market becomes more efficient, with prices better reflecting the true value of assets based on underlying data and sentiment.
  
- **Risk Management**: For institutional investors and large-scale traders, predictions are essential for managing exposure to market risks. Reliable forecasts enable better hedging strategies and risk mitigation.
  
- **Innovation and Development**: Crypto price prediction models drive innovation in financial technology, contributing to the development of more sophisticated trading algorithms and automated systems.

### Real-World Necessity of the CryptoFlow Prediction Subnet

In the rapidly evolving world of cryptocurrencies, precise price predictions are not just beneficial—they are essential for several key reasons:

- **Market Stability**: Reliable price predictions can contribute to greater market stability by helping traders and investors anticipate and react to market movements more effectively. This can lead to smoother market operations and reduce the impact of sudden price swings.

- **Enhanced Trading Strategies**: Traders and institutional investors use predictions to develop and implement trading strategies. Accurate forecasting helps them optimize their entry and exit points, leading to more effective trading strategies and potentially higher returns.

- **Economic Impact**: Cryptocurrencies are increasingly becoming a part of mainstream financial systems and investment portfolios. Accurate price predictions are crucial for economic planning, portfolio management, and investment decisions at both individual and institutional levels.

- **Informed Regulation**: Governments and financial regulators can use price predictions to better understand market dynamics and develop more informed regulatory policies. This can lead to a more balanced and fair regulatory environment.

- **Innovation in Financial Services**: The development of predictive models drives innovation in financial services, leading to new tools and platforms that can offer better insights and more efficient solutions for managing cryptocurrency investments.

## Features

- **Decentralized Predictions**: Operates within the Bittensor network, ensuring a fully decentralized and secure environment for generating crypto price predictions.
- **Multi-Currency Support**: Predicts prices for multiple cryptocurrencies, including TAO, BTC, ETH, and others.
- **Cutting-Edge Models**: Utilizes advanced AI models fine-tuned for the unique challenges of crypto price forecasting.
- **Real-Time Insights**: Provides up-to-date predictions that can be used for trading and investment strategies.

## Getting Started

### Prerequisites

To run the CryptoFlow Prediction Subnet, you'll need the following:

- **Bittensor CLI**: Ensure you have the Bittensor CLI installed. Follow the [official documentation](https://github.com/opentensor/bittensor) to get started.
- **Python 3.8+**: The subnet's code is written in Python, so you'll need Python 3.8 or higher.
- **Dependencies**: Install the necessary Python packages by running:

  ```bash
  pip install -r requirements.txt
  ```

### Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/mltrev23/crypto-prediction-subnet.git
cd crypto-prediction-subnet
```

### Running the Subnet

#### For Validators

To start the validator, use the following command:

```bash
python3 neurons/validator.py --subtensor.network <network> --netuid <netuid> --wallet.name <wallet_name> --wallet.hotkey <validator_hotkey> --logging.debug --logging.trace
```

- Replace `<network>` with the name of the network you are connecting to (e.g., `test`).
- Replace `<netuid>` with the specific network UID you are targeting.
- Replace `<wallet_name>` with the name of your wallet.
- Replace `<validator_hotkey>` with the hotkey associated with your validator.

Example:

```bash
python3 neurons/validator.py --subtensor.network test --netuid 205 --wallet.name crypto_prediction --wallet.hotkey validator1 --logging.debug --logging.trace
```

#### For Miners

To start the miner, use the following command:

```bash
python3 neurons/miner.py --subtensor.network <network> --netuid <netuid> --wallet.name <wallet_name> --wallet.hotkey <miner_hotkey> --logging.debug --logging.trace --model <model_path> --axon.port <port>
```

- Replace `<network>` with the name of the network you are connecting to (e.g., `test`).
- Replace `<netuid>` with the specific network UID you are targeting.
- Replace `<wallet_name>` with the name of your wallet.
- Replace `<miner_hotkey>` with the hotkey associated with your miner.
- Replace `<model_path>` with the path to the model you are using (e.g., `models/base_lstm.h5`).
- Replace `<port>` with the port number your axon is using.

Example:

```bash
python3 neurons/miner.py --subtensor.network test --netuid 205 --wallet.name crypto_prediction --wallet.hotkey miner1 --logging.debug --logging.trace --model models/base_lstm.h5 --axon.port 8092
```

## Contribution

We welcome contributions from the community! If you'd like to contribute to the CryptoFlow Prediction Subnet, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Bittensor Team**: For developing and maintaining the Bittensor network.
- **Contributors**: Thanks to all contributors who have helped shape this project.

## Contact

For any questions or support, please reach out to the project maintainer at [trevor.dev23@gmail.com](mailto:trevor.dev23@gmail.com).

---

*Happy Predicting!*
