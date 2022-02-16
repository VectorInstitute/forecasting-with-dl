# Multivariate Forecasting Demo Notebooks

This directory contains a number of Jupyter notebooks designed to help you experiment with a variety of forecasting problems.

## Dependencies

In testing, we have identified version incompatibilities with Pandas. Please ensure your environment is configured with `pandas==1.3.0`. You can check your version of Pandas using the command `pip show pandas`. If your version of Pandas is not correct, please install the required version using the command `pip install pandas=1.3.0`.

## Demos

### Baselines, Prophet, and NeuralProphet

Prophet and NeuralProphet applied to exchange rate forecasting. One configuration of Prophet and three configurations of NeuralProphet are considered. Uses a conventional data splitting approach (80% train, 20% test). Applies a collection of four evaluation metrics (MSE, rMSE, MAE, MAPE) in a consistent fashion over all experiments.

### PyTorch Forecasting

N-BEATS, DeepAR, and Temporal Fusion Transformer applied to exchange rate forecasting.

### Rolling Cross-Validation

Alternative data splitting and experimental design using rolling cross-validation and NeuralProphet. Applied to exchange rate forecasting.
