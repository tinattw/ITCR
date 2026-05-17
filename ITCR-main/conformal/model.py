from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (before sigmoid)
        # targets: binary labels (0 or 1)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if target=1, else 1-p

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class FocalLossModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FocalLossModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FocalLossClassifier:
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, 
                 alpha=0.25, gamma=2.0, lr=0.001, batch_size=32, 
                 epochs=100, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model = FocalLossModel(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def classes_(self):
        return np.array([0, 1])