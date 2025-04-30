import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1) Define your model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # time-distributed linear: applies to each time step
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)         # out: [batch, seq_len, hidden_dim]
        out = self.fc(out)            # out: [batch, seq_len, output_dim]
        return out

# 2) Hyperparameters
input_dim   = 5
hidden_dim  = 32
output_dim  = 2
num_layers  = 1
batch_size  = 64
lr          = 1e-3
num_epochs  = 20

# 3) Dummy data (replace with your tensors)
X = torch.randn(1000, 11, input_dim)   # your features
Y = torch.randn(1000, 11, output_dim)  # your targets

# 4) DataLoader
dataset = TensorDataset(X, Y)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5) Instantiate model, loss, optimizer
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model    = Seq2SeqLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion= nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters(), lr=lr)

# 6) Training loop
for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # [B,11,5], [B,11,2]
        optimizer.zero_grad()

        preds = model(xb)                      # [B,11,2]
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch:02d} — loss: {epoch_loss:.4f}")
