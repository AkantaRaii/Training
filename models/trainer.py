import torch
import numpy as np
seed = 42
np.random.seed(seed)
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, val_loader, epochs, patience, model_path):
        best_loss = float('inf')
        patience_counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, verbose=True)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.transpose(0, 1).to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X.size(1)

            train_loss /= len(train_loader.dataset)

            val_loss = self.evaluate(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.transpose(0, 1).to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                val_loss += loss.item() * X.size(1)
        return val_loss / len(loader.dataset)