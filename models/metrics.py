import numpy as np
from sklearn.preprocessing import StandardScaler

seed = 42
np.random.seed(seed)
class Metrics:
    def __init__(self,predicted_train,actuals_train,predicted_test,actuals_test,scaler,scaled_data):
        print(scaled_data)
        
        self.actuals_test_rescaled = scaler.inverse_transform(np.concatenate((actuals_test, np.zeros((actuals_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
        self.predictions_test_rescaled = scaler.inverse_transform(np.concatenate((predicted_test, np.zeros((predicted_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
        self.actuals_train_rescaled = scaler.inverse_transform(np.concatenate((actuals_train, np.zeros((actuals_train.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
        self.predictions_train_rescaled = scaler.inverse_transform(np.concatenate((predicted_train, np.zeros((predicted_train.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
        print(self.actuals_test_rescaled[0],self.predictions_test_rescaled[0])

    def thresold_accuracy(self):
        # Define the threshold (e.g., 5% tolerance)
        threshold = 0.05
        
        # Calculate the percentage error for each prediction
        percentage_errors = np.abs((self.actuals_test_rescaled - self.predictions_test_rescaled) / self.actuals_test_rescaled)
        
        # Calculate accuracy as the percentage of predictions within the threshold
        accuracy = np.mean(percentage_errors <= threshold) * 100
        return accuracy
    def train_mape(self):
        actuals = [1e-10 if value == 0 else value for value in self.actuals_train_rescaled]


        # Calculate the absolute percentage errors
        mape = np.mean(np.abs((actuals - self.predictions_train_rescaled) / actuals)) * 100
        return mape
    def test_mape(self):
        actuals = [1e-10 if value == 0 else value for value in self.actuals_test_rescaled]


        # Calculate the absolute percentage errors
        mape = np.mean(np.abs((actuals - self.predictions_test_rescaled) / actuals)) * 100
        return mape