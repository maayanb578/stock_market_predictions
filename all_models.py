import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import tensorflow as tf

keras = tf.keras

def plot_series(time, series, format="-", start=0, end=None, label=None):
    """[Plot the series data over a time range]
    Args:
        time (data range): [The entire time span of the data in range format]
        series ([integers]): [Series value corresponding to its point on the time axis]
        format (str, optional): [Graph type]. Defaults to "-".
        start (int, optional): [Time to start time series data]. Defaults to 0.
        end ([type], optional): [Where to stop time data]. Defaults to None.
        label ([str], optional): [Label name of series]. Defaults to None.
    """
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def ma20(df):
    window = 20

    # Create a moving average over the entire dataset
    moving_avg = df['Close'].rolling(window=window).mean()
    test_split_date = '2018-01-02'
    test_split_index = np.where(df.Date == test_split_date)[0][0]
    x_test = df.loc[df['Date'] >= test_split_date]['Close']

    # Slice the moving average on the forecast
    moving_avg_forecast = moving_avg.values[test_split_index - window:df.index.max() - window + 1]


    # create the figure for presenting the predection result
    plt.figure(figsize=(10, 6))
    plot_series(x_test.index, x_test, label="Series")
    plot_series(x_test.index, moving_avg_forecast, label="Moving average (20 days)")
    plt.ylabel('Dollars $')
    plt.xlabel('Timestep in Days')
    plt.title('SMA20 vs Actual')
    plt.show()

    # calculate the mean error for the ML model
    ma_20 = keras.metrics.mean_absolute_error(x_test, moving_avg_forecast).numpy()
    return ma_20

def naive(df):

    df['Date'] = pd.to_datetime(df['Date'])
    df.tail(100)
    df.dropna(inplace=True)

    series = df['Close']
    # Create train data set from the given date
    train_split_date = '2002-02-25'
    train_split_index = np.where(df.Date == train_split_date)[0][0]
    x_train = df.loc[df['Date'] <= train_split_date]['Close']

    # Create test data set from the given date
    test_split_date = '2021-05-10'
    test_split_index = np.where(df.Date == test_split_date)[0][0]
    x_test = df.loc[df['Date'] >= test_split_date]['Close']

    # Create valid data set
    x_valid = df.loc[(df['Date'] < test_split_date) & (df['Date'] > train_split_date)]['Close']

    # set style of charts
    sns.set(style="darkgrid")
    plt.rcParams['figure.figsize'] = [10, 10]

    # Create a plot showing the split of the train, valid, and test data
    plt.plot(x_train, label='Train')
    plt.plot(x_valid, label='Validate')
    plt.plot(x_test, label='Test')
    plt.title('Train Valid Test Split of Data')
    plt.ylabel('Dollars $')
    plt.xlabel('Timestep in Days')
    plt.legend()
    print(x_train.index.max(), x_valid.index.min(), x_valid.index.max(), x_test.index.min(), x_test.index.max())

    plt.show()

    model_mae_scores = {}

    # Plot chart with all details untouched
    plot_series(time=df.index, series=df['Close'], label='Apple Close Price')
    plt.ylabel('Dollars $')
    plt.xlabel('Timestep in Days')
    plt.title('Price History of Apple Jan-1993 to Dec-2020')

    naive_forecast = series[test_split_index - 1:-1]

    plt.figure(figsize=(10, 6))
    plot_series(x_test.index, x_test, label="Actual")
    plot_series(x_test.index, naive_forecast, label="Forecast")
    plt.ylabel('Dollars $')
    plt.xlabel('Timestep in Days')
    plt.title('Naive Forecast vs Actual')
    plt.show()

    # Alternative way to show MAE to stay consistent with what we will be doing later
    naive_forecast_mae = keras.metrics.mean_absolute_error(x_test, naive_forecast).numpy()
    model_mae_scores['naive'] = naive_forecast_mae

    # view the dictionary of mae scores
    print(model_mae_scores)
    return model_mae_scores

    # Show first 3 values of our forecast


def decision_tree(df):

    # print the shaped data for developer view
    print(df.shape)
    df.head()

    df = df.reset_index(drop=True)
    df.head()

    col_list = df.columns.tolist()
    print(col_list)


    # Create plot for the initial data
    sns.set()
    plt.figure(figsize=(10, 6))
    plt.title("Stock Price")
    plt.xlabel("Days")
    plt.ylabel("Close Price USD ($)")
    plt.plot(df["Close"])
    plt.show()


    # shift(remove) the data for the last 30 days for the prediction results
    futureDays = 30
    df["Prediction"] = df[["Close"]].shift(-futureDays)
    print(df.head())
    print(df.tail())

    x = np.array(df.drop(["Prediction"], 1))[:-futureDays]
    y = np.array(df["Prediction"])[:-futureDays]

    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)


    keras = tf.keras
    tree = DecisionTreeRegressor().fit(xtrain, ytrain)

    # creating the Linear Regression model
    from sklearn.linear_model import LinearRegression
    linear = LinearRegression().fit(xtrain, ytrain)


    xfuture = df.drop(["Prediction"], 1)[:-futureDays]
    xfuture = xfuture.tail(futureDays)
    xfuture = np.array(xfuture)
    print(xfuture)

    treePrediction = tree.predict(xfuture)
    print("Decision Tree prediction =",treePrediction)


    linearPrediction = linear.predict(xfuture)
    print("Linear regression Prediction =",linearPrediction)
    #

    predictions = treePrediction
    valid = df[x.shape[0]:]
    valid["Predictions"] = predictions
    plt.figure(figsize=(10, 6))
    plt.title("'s Stock Price Prediction Model(Decision Tree Regressor Model)")
    plt.xlabel("Days")
    plt.ylabel("Close Price USD ($)")
    plt.plot(df["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Original", "Valid", "Predictions"])
    plt.show()


    y_pred = tree.predict(xtest)

    # Compute mse_dt
    mse_dt = MSE(ytest, y_pred)

    # Compute rmse_dt
    rmse_dt = mse_dt**(1/2)

    # Print rmse_dt
    print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)
    df_log.head()
    return rmse_dt

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def window_dataset(series, window_size, batch_size=128,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def dense(df):

    # Save target series
    series = df['Close']

    # Create train data set
    train_split_date = '2014-12-31'
    train_split_index = np.where(df.Date == train_split_date)[0][0]
    x_train = df.loc[df['Date'] <= train_split_date]['Close']

    # Create test data set
    test_split_date = '2021-06-01'
    test_split_index = np.where(df.Date == test_split_date)[0][0]
    x_test = df.loc[df['Date'] >= test_split_date]['Close']

    # Create valid data set
    valid_split_index = (train_split_index.max(), test_split_index.min())
    x_valid = df.loc[(df['Date'] < test_split_date) & (df['Date'] > train_split_date)]['Close']

    plt.plot(x_train, label='Train')
    plt.plot(x_valid, label='Validate')
    plt.plot(x_test, label='Test')
    plt.legend()
    print(x_train.index.max(), x_valid.index.min(), x_valid.index.max(), x_test.index.min(), x_test.index.max())
    plt.show()

    # Reshape values
    x_train_values = x_train.values.reshape(-1, 1)
    x_valid_values = x_valid.values.reshape(-1, 1)
    x_test_values = x_test.values.reshape(-1, 1)

    #  Create Scaler Object
    x_train_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit x_train values
    normalized_x_train = x_train_scaler.fit_transform(x_train_values)

    # Fit x_valid values
    normalized_x_valid = x_train_scaler.transform(x_valid_values)

    # Fit x_test values
    normalized_x_test = x_train_scaler.transform(x_test_values)

    # All values normalized to training data
    df_normalized_to_traindata = x_train_scaler.transform(series.values.reshape(-1, 1))

    # Example of how to iverse
    # inversed = scaler.inverse_transform(normalized_x_train).flatten()

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    window_size = 20
    train_set = window_dataset(normalized_x_train.flatten(), window_size)

    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-7 * 10 ** (epoch / 20))
    optimizer = keras.optimizers.Nadam(lr=1e-7)
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set, epochs=50, callbacks=[lr_schedule])

    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-7, 1, 0, .1])

    # Clear back end
    keras.backend.clear_session()

    # Ensure reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Set Window Size
    window_size = 30
    train_set = window_dataset(normalized_x_train.flatten(), window_size)
    valid_set = window_dataset(normalized_x_valid.flatten(), window_size)

    # Build 2 layer model with 10 neurons each and 1 output layer
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])

    # Set optimizer
    optimizer = keras.optimizers.Nadam(lr=1e-2)
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    # Set early Stopping
    early_stopping = keras.callbacks.EarlyStopping(patience=20)

    # create save points for best model
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "my_checkpoint", save_best_only=True)

    # Fit model
    history = model.fit(train_set, epochs=30,
                        validation_data=valid_set,
                        callbacks=[early_stopping, model_checkpoint])

    model = keras.models.load_model("my_checkpoint")

    dense_forecast = model_forecast(model, df_normalized_to_traindata.flatten()[x_test.index.min() - window_size:-1],
                                    window_size)[:, 0]

    df_normalized_to_traindata.flatten().shape

    # Undo the scaling
    dense_forecast = x_train_scaler.inverse_transform(dense_forecast.reshape(-1, 1)).flatten()
    dense_forecast.shape

    # set style of charts
    sns.set(style="darkgrid")
    plt.rcParams['figure.figsize'] = [10, 10]

    plt.figure(figsize=(10, 6))
    plt.title('Fully Dense Forecast')
    plt.ylabel('Dollars $')
    plt.xlabel('Timestep in Days')
    plot_series(x_test.index, x_test)
    plot_series(x_test.index, dense_forecast)

    plt.show()

    mean_err = keras.metrics.mean_absolute_error(x_test, dense_forecast).numpy()
    return mean_err
