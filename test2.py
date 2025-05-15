import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1) Generate synthetic data
batch_size = 32
sequence_length = 50
dim = 3

t = torch.linspace(0, 4 * torch.pi, sequence_length)
base = torch.stack([
    torch.sin(t + phase) for phase in torch.linspace(0, torch.pi, dim)
], dim=1)  # (sequence_length, dim)

data = base.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, sequence_length, dim)
noise = 0.1 * torch.randn_like(data)
data_noisy = data + noise  # (batch_size, sequence_length, dim)
import numpy as np
# 2) Dataset and DataLoader
dataset = TensorDataset(data_noisy, data_noisy)   # autoencoder style
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3) Define model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, S, dim)
        out, _ = self.lstm(x)         
        return self.fc(out)           # (B, S, dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqLSTM(input_dim=dim, hidden_dim=16, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 4) Training loop
num_epochs = 50
loss_history = []

model.train()
for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(dataset)
    loss_history.append(epoch_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}")

# 5) Plot loss
plt.plot(loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
