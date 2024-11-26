import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models.transformer import Transformer
from models.metrics import Metrics
from models.dataprocessor import LassoDataProcessor
from models.trainer import Trainer
from models.tester import Tester
    



def lasso_main(stock,sequence_length ):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    file_path = f'../data/lasso_data/l{stock}.csv'
    model_path=f'../best_model/bestmodel_lasso/{stock}.pth'
    
    processor = LassoDataProcessor(file_path, sequence_length)
    scaled_data = processor.load_and_preprocess()
    X, y = processor.create_sequences(scaled_data)
    train_size=int(len(X)*0.8)
    X_train=X[:train_size]
    X_test=X[train_size:]
    y_train=y[:train_size]
    y_test=y[train_size:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=60, shuffle=True,pin_memory=True if device.type=='cuda'else False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=60, shuffle=False,pin_memory=True if device.type=='cuda'else False)

    feature_count = X_train.shape[2]
    model = Transformer(features=feature_count, d_model=512, head=8, dropout=0.3, num_layers=6)
    criterion = torch.nn.MSELoss()
    weight_decay = 1e-5  # Regularization to avoid overfitting
    lr=0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.train(train_loader, test_loader, epochs=100, patience=20, model_path=model_path)

    tester=Tester(model,criterion, device)
    predicted_train,actual_train,predicted_test,actual_test=tester.test(train_loader,test_loader,model_path=model_path)
    metrics=Metrics(predicted_train,actual_train,predicted_test,actual_test,processor.scaler,scaled_data)
    torch.cuda.empty_cache()
    return metrics.thresold_accuracy(),metrics.train_mape(),metrics.test_mape()

if __name__ == "__main__":
    import csv

    stocks = [
        "ADBL", "CZBIL", "EBL", "GBIME", "HBL", "KBL", "MBL", "NABIL", "NBL",
        "NICA", "NMB", "PCBL", "SANIMA", "SBI", "SBL", "SCB", "PRVU", "NIMB", "LSL"
    ]

    with open('resultsasdas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Stock", "Sequence", "Accuracy", "Train MAPE", "Test MAPE"])  # Header
        for stock in stocks:
            for sequence in range(3, 11):
                print(f'<======================={stock}====================>')
                accuracy, train_mape, test_mape = lasso_main(stock, sequence)
                writer.writerow([stock, sequence, accuracy, train_mape, test_mape])


