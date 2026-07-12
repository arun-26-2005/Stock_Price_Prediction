import os
import json

DATASETS = {
    "TCS": "data/TCS_data.csv",
    "RELIANCE": "data/Reliance_data.csv",
    "INFY": "data/INFY_data.csv"
}

def load_custom_datasets():
    # Load custom datasets from JSON file
    registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_stocks.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                custom = json.load(f)
                DATASETS.update(custom)
        except Exception as e:
            print(f"Error loading custom stocks registry: {e}")

load_custom_datasets()

MODELS = {
    "LSTM": "lstm",
    "BILSTM": "bilstm",
    "MTRAN": "mtran",
    "CNN_BILSTM": "cnn_bilstm",
    "CNN_BILSTM_AM": "cnn_bilstm_am",
    "MTRAN_TCN": "mtran_tcn",
    "BILSTM_TCN": "bilstm_tcn",
    "BILSTM_MTRAN": "bilstm_mtran",
    "HYBRID": "hybrid_bilstm_mtran_tcn"
}