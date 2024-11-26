from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
class PcaDataProcessor:
    def __init__(self, file_path, sequence_length):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        data = pd.read_csv(self.file_path)
        close=data['close'].values.reshape(-1,1)
        scaled_data=data.drop(columns=['date','close'])
        self.close = self.scaler.fit_transform(close)
        scaled_data['close_scaled']=np.squeeze(self.close)
        return scaled_data

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.sequence_length):
            x = data[i:i + self.sequence_length]
            y = self.close[i + self.sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.squeeze(np.array(ys))
    
class LassoDataProcessor:
    def __init__(self, file_path, sequence_length):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        data = pd.read_csv(self.file_path).drop(columns=['date'])
        scaled_data = self.scaler.fit_transform(data)
        self.close_index = data.columns.get_loc('close')
        return scaled_data

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.sequence_length):
            x = data[i:i + self.sequence_length]
            y = data[i + self.sequence_length, self.close_index]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)