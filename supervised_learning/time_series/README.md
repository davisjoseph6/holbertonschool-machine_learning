# Bitcoin Price Forecasting with RNN

This project demonstrates how to forecast Bitcoin prices using a Recurrent Neural Network (RNN). The process involves loading and preprocessing raw Bitcoin price data, building an RNN model for time series forecasting, and visualizing the training process.

---

## Directory Overview

### Files and Scripts

1. **`preprocess_data.py`**
   - **`load_and_preprocess(file_path)`**
     - Loads and preprocesses raw Bitcoin data.
     - **Steps**:
       - Loads the raw CSV data containing Bitcoin prices.
       - Drops missing values and selects relevant features (`Timestamp` and `Close`).
       - Resamples the data to hourly intervals.
       - Normalizes the 'Close' prices.
       - Creates rolling windows of 24 hours to prepare data for time series forecasting.
       - Saves the processed data to `X.npy` (features) and `y.npy` (targets).

2. **`forecast_btc.py`**
   - **`create_rnn_model(input_shape)`**
     - Defines an RNN model for Bitcoin price forecasting using `SimpleRNN` and a `Dense` layer.
     - **Inputs**:
       - `input_shape`: Shape of the input data (number of time steps, number of features).
     - **Outputs**:
       - A compiled RNN model.
   
   - **`plot_training_history(history)`**
     - Plots and saves the training and validation loss over epochs.
     - **Inputs**:
       - `history`: The training history object from `model.fit()`.
     - **Outputs**:
       - A plot of training and validation loss.
   
   - **`main()`**
     - Loads the preprocessed data from `X.npy` and `y.npy`.
     - Splits the data into training and validation sets (80% training, 20% validation).
     - Creates a `tf.data.Dataset` for efficient batch processing.
     - Initializes the RNN model and trains it using the training data.
     - Plots the training/validation loss and saves the model to `btc_rnn_model.h5`.

---

## Data Files

- **`coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip`**
  - Raw Bitcoin price data from Coinbase (1-minute intervals).
  - **Columns**:
    - `Timestamp`: Unix timestamp of the price.
    - `Close`: Closing price of Bitcoin at each timestamp.

- **`bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv.zip`**
  - Raw Bitcoin price data from Bitstamp (1-minute intervals).
  - Used similarly to the Coinbase data in preprocessing.

- **`X.npy` & `y.npy`**
  - Processed data files containing:
    - `X`: Features for training (24-hour rolling windows).
    - `y`: Target values (next hour's price).

---

## Training and Evaluation

- **Model Architecture**:
  - The RNN model consists of:
    - A `SimpleRNN` layer with 50 units and ReLU activation.
    - A `Dense` layer to output the predicted next price.
  
- **Loss Function**:
  - Mean Squared Error (MSE) is used to evaluate the performance of the model.

- **Training**:
  - The model is trained for 10 epochs with a batch size of 32.
  - Training and validation loss are visualized and saved as `training_validation_loss.png`.

- **Model Saving**:
  - The trained model is saved as `btc_rnn_model.h5` for future use.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib

---

## How to Run

1. **Preprocess the Data**:
   - First, run `preprocess_data.py` to load and preprocess the raw Bitcoin data.
     ```bash
     python3 preprocess_data.py
     ```

2. **Train the Model**:
   - After preprocessing, run `forecast_btc.py` to train the RNN model.
     ```bash
     python3 forecast_btc.py
     ```

3. **Results**:
   - Training and validation loss will be saved as `training_validation_loss.png`.
   - The trained model will be saved as `btc_rnn_model.h5`.

---

## Applications

- **Bitcoin Price Prediction**: Predict the next hour's price of Bitcoin based on historical data.
- **Time Series Forecasting**: Generalizable to other time series forecasting tasks.
- **Financial Market Prediction**: Can be adapted to predict stock prices or other financial instruments.

---

## References

- [Predicting Bitcoin Price Using RNN: A Deep Dive into Time Series](https://www.linkedin.com/pulse/predicting-bitcoin-price-using-rnn-deep-dive-time-series-davis-joseph-kkhte/)

---

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

