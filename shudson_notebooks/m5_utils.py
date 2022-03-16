import pandas as pd
from sklearn.metrics import mean_squared_error

def m5_evaluation(forecast, truth=None, method='rmse'):
    
    if truth is None:
        try:
            truth = pd.read_csv('m5_truth.csv').set_index('id')
        except:
            raise FileNotFoundError('m5_truth.csv could not be found. It is required when the truth dataframe is not provided.')

    #
    # DFs are transposed so mean_squared_error will:
    #   1. Calculates an error for each product (instead of for each forecast day)
    #   2. Weight the product errors (uniform for RMSE)
    #
    truth_sorted_tp = truth.sort_index().transpose(copy=True)
    forecast_sorted_tp = forecast.sort_index().transpose(copy=True)
    
    forecast_score = None
    
    if method.upper() == 'RMSE':
        forecast_score = mean_squared_error(truth_sorted_tp, forecast_sorted_tp, squared=False)
        
    elif method.upper() == 'WRMSSE':
        # TODO: consider implementing this evaluation metric as it was used for the Kaggle competition.
        # Note that mean_squared_error can provide the error for each product (multioutput='raw_values').
        # (Useful for RMSSE?)
        # forecast_score = mean_squared_error(truth_sorted_tp, forecast_sorted_tp, squared=False, multioutput='raw_values')
        raise ValueError('The WRMSSE metric is not yet implemented')
                
    else:
        raise ValueError(f'{method.upper()} is not a supported evaluation metric')
        
    return forecast_score
