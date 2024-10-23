import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('Dow Jones Industrial Average Historical Data.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')

def calc_percent_change(dataset):
    dataset['Price'] = dataset['Price'].str.replace(',', '')
    price = dataset['Price']
    np_price = np.array(price, dtype=np.float32)
    percent_change = np.zeros(dataset.shape[0], dtype=np.float32)
    percent_change[1:] = ((np_price[1:] - np_price[:-1]) / np_price[:-1]) * 100

    return percent_change

def create_dataset_for_LSTM(dataset, lookback):
    X,y = list(), list()
    for i in range(len(dataset)-lookback):
        X.append(dataset[i:i+lookback])
        y.append(dataset[i+lookback])

    return np.array(X), np.array(y)

lookback = 4
percent_change_data = calc_percent_change(dataset)
X, y = create_dataset_for_LSTM(percent_change_data, lookback)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val  = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=lookback, hidden_size=100, num_layers=1, batch_first=True)
        self.linear = nn.Linear(100, 1)

    def forward(self, data):
        out, _ = self.lstm(data)
        out = self.linear(out)
        return out

model = LSTM()
min_val_loss = float('inf')
early_stop_patience = 10
patience_counter = 1
learning_rate = 0.05
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 500
for epoch in range(epochs):
    model.train()

    train_loss = 0.0
    for X_train_batch, y_train_batch in train_loader:
        optimizer.zero_grad()

        y_val_pred = model(X_train_batch)
        loss = criterion(y_val_pred, y_train_batch)
        train_loss = loss.item()

        loss.backward()
        
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = model(X_val_batch)
            loss = criterion(y_val_pred, y_val_batch)

            val_loss += loss.item()
        
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch}, Train loss: {avg_train_loss:.2f}, Val loss: {avg_val_loss:.2f} ')

    if val_loss < min_val_loss:
        patience_counter = 1
        min_val_loss = val_loss

    else:
        patience_counter += 1
    
    if patience_counter > early_stop_patience:
        print('Early Stopping!!!')
        break

with torch.no_grad():
    train_plot = np.full(len(X_train_tensor), np.nan)
    test_plot = np.full(len(X_test_tensor), np.nan)
    train_plot[:X_train_tensor.shape[0]] = model(X_train_tensor).view(-1)
    test_plot = model(X_test_tensor).view(-1)

# plot
plt.figure(figsize=(100, 7))
plt.plot(dataset['Date'], percent_change_data, label='Percent Change', color='b')  
plt.plot(dataset['Date'][:X_train_tensor.shape[0]], train_plot, c='r', label='Train Predictions')
plt.plot(dataset['Date'][X_train_tensor.shape[0] + X_val_tensor.shape[0] + lookback:], test_plot, c='g', label='Test Predictions')
plt.xlim(dataset['Date'].iloc[0], dataset['Date'].iloc[-1])
plt.xlabel('Date')
plt.ylabel('Percent Change')
plt.title('LSTM Predictions of Percent Change')
plt.legend()
plt.show()