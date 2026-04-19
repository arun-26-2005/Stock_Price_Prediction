import argparse
import pandas as pd
import torch
from config import DATASETS, MODELS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm


def create_sequences(data, window=10):
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][0])

    return np.array(X), np.array(y)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="TCS")
parser.add_argument("--model", type=str, default="HYBRID")
args = parser.parse_args()
dataset_name = args.dataset.upper()

if dataset_name not in DATASETS:
    raise ValueError(f"Dataset must be one of {list(DATASETS.keys())}")

file_path = DATASETS[dataset_name]
df = pd.read_csv(file_path)

#Preprocessing

df.columns=['Date','Close','High','Low','Open','Volume']
df=df.drop([0,1]).reset_index(drop=True)
df = df.dropna()
df = df.reset_index(drop=True)
df=df.sort_values('Date')
df['MA5']=df['Close'].rolling(5).mean()
df['MA10']=df['Close'].rolling(10).mean()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()

split_date = '2023-01-01'

train_df = df[df['Date'] < split_date]
test_df  = df[df['Date'] >= split_date]

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_df.drop(columns=['Date']))
test_scaled = scaler.transform(test_df.drop(columns=['Date']))
train_scaled = np.nan_to_num(train_scaled)
test_scaled = np.nan_to_num(test_scaled)

window_size = 10

X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


#Dataset and Dataloader
from torch.utils.data import Dataset, DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Model, Loss, Optimizer
from models.hybrid_bilstm_mtran_tcn import BiLSTM_MTRAN_TCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = args.model.upper()

if model_name not in MODELS:
    raise ValueError(f"Model must be one of {list(MODELS.keys())}")

if model_name == "LSTM":
    from models.lstm import LSTMModel
    model = LSTMModel(input_size=X_train.shape[2])
    epochs = 1000
    lr = 0.0001

elif model_name == "BILSTM":
    from models.bilstm import BiLSTMModel
    model = BiLSTMModel(input_size=X_train.shape[2])
    epochs = 600
    lr = 0.0001

elif model_name == "MTRAN":
    from models.mtran import MTRAN
    model = MTRAN(input_size=X_train.shape[2])
    epochs = 300
    lr = 1e-5

elif model_name == "CNN_BILSTM":
    from models.cnn_bilstm import CNN_BiLSTM
    model = CNN_BiLSTM(input_size=X_train.shape[2])
    epochs = 400
    lr = 0.0001


elif model_name == "CNN_BILSTM_AM":
    from models.cnn_bilstm_am import CNN_BiLSTM_AM
    model = CNN_BiLSTM_AM(input_size=X_train.shape[2])
    epochs = 200
    lr = 0.0001

elif model_name == "MTRAN_TCN":
    from models.mtran_tcn import MTRAN_TCN
    model = MTRAN_TCN(input_size=X_train.shape[2])
    epochs = 1e-5
elif model_name == "BILSTM_TCN":
    from models.bilstm_tcn import BiLSTM_TCN
    model = BiLSTM_TCN(input_size=X_train.shape[2])
    epochs = 200
    lr = 0.00001
elif model_name == "BILSTM_MTRAN":
    from models.bilstm_mtran import BiLSTM_MTRAN
    model = BiLSTM_MTRAN(input_size=X_train.shape[2])
    epochs = 200
    lr = 0.00001

elif model_name == "HYBRID":
    from models.hybrid_bilstm_mtran_tcn import BiLSTM_MTRAN_TCN
    model = BiLSTM_MTRAN_TCN(input_size=X_train.shape[2])
    epochs = 600
    lr = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)   
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    # tqdm for batches
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for X_batch, y_batch in loop:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")



# Evaluation

model.eval()

predictions = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        predictions.extend(preds.squeeze().cpu().numpy())

from utils.evaluation import evaluate_model


pred_array = np.zeros((len(predictions), train_scaled.shape[1]))
pred_array[:, 0] = predictions
pred_original = scaler.inverse_transform(pred_array)[:, 0]

actual_array = np.zeros((len(y_test), train_scaled.shape[1]))
actual_array[:, 0] = y_test.cpu().numpy()
actual_original = scaler.inverse_transform(actual_array)[:, 0]

evaluate_model(
    actual_original,
    pred_original,
    scaler,
    y_test,
    train_scaled,
    test_df=test_df,
    window_size=window_size
)
