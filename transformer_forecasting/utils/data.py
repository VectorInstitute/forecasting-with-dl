import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def split_data(data, dates, train_perc, test_perc):
    n_samples = data.shape[0]
    num_train = int(n_samples * train_perc)
    num_test = int(n_samples * test_perc)
    num_val = n_samples - num_train - num_test

    # Sample train set
    train_data = data.iloc[0:num_train, :]
    train_dates = pd.DatetimeIndex(dates.iloc[0:num_train])
    train_time_feat = get_day_features(train_dates)
    scaled_train_data = scale_data(train_data)

    # Sample validation set
    val_data = data.iloc[num_train: num_train+num_val, :]
    val_dates = pd.DatetimeIndex(dates.iloc[num_train: num_train+num_val])
    val_time_feat = get_day_features(val_dates)
    scaled_val_data = scale_data(val_data)

    # Sample Test Set
    test_data = data.iloc[num_train+num_val:, :]
    test_dates = pd.DatetimeIndex(dates.iloc[num_train+num_val:])
    test_time_feat = get_day_features(test_dates)
    scaled_test_data = scale_data(test_data)

    return (scaled_train_data, train_time_feat), (scaled_val_data, val_time_feat), (scaled_test_data, test_time_feat)

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data.values)
    scaled_data = scaler.transform(data.values)
    return scaled_data

def get_day_features(datetime_index):
    # Encode day of week as value between [-.5, .5]
    dow_feat = datetime_index.dayofweek / 6.0 - 0.5

    # Encode day of month as value between [-.5, .5]
    dom_feat = (datetime_index.day - 1) / 30.0 - 0.5

    # Encode day of year as value between [-.5, .5]
    doy_feat = (datetime_index.dayofyear - 1) / 365.0 - 0.5

    # Stack into time feature matrix
    time_feat = np.stack((dow_feat, dom_feat, doy_feat), axis=0).transpose(1, 0)

    return time_feat