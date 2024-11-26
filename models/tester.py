import torch
import numpy as np
seed = 42
np.random.seed(seed)
class Tester:
    def __init__(self, model,criterion, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
    def test(self,train_loader,test_loader,model_path):
        self.model.load_state_dict(torch.load(model_path))
        predicted_test=[]
        actual_test=[]
        predicted_train=[]
        actual_train=[]
        self.model.eval()
        test_loss=0
        with torch.no_grad():
            for X_test,y_test in test_loader:
                X_test,y_test=X_test.transpose(0,1).to(self.device),y_test.to(self.device)
                test_output=self.model(X_test)
                loss=self.criterion(test_output,y_test)
                test_loss+=loss.item()*len(X_test)
                predicted_test.extend(test_output.detach().cpu().numpy())
                actual_test.extend(y_test.cpu().numpy())
            for X_train,y_train in train_loader:
                X_train,y_train=X_train.transpose(0,1).to(self.device),y_train.to(self.device)

                train_output=self.model(X_train)
                # print(test_output.shape)

                predicted_train.extend(train_output.detach().cpu().numpy())
                actual_train.extend(y_train.cpu().numpy())

        test_loss/=len(test_loader.dataset)
        predicted_train=np.array(predicted_train)
        actual_train=np.array(actual_train)
        predicted_test=np.array(predicted_test)
        actual_test=np.array(actual_test)
        
        return predicted_train,actual_train,predicted_test,actual_test