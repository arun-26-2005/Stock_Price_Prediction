---
title: Stock Price Prediction Backend
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Stock Price Prediction using Deep Learning

## Overview
This project focuses on predicting stock prices using multiple deep learning architectures. It compares traditional sequence models with advanced hybrid architectures to capture complex temporal patterns in financial time series data.

## Dataset
* **Timeframe:** Historical stock data (2015–2025)
* **Source:** Yahoo Finance
* **Stocks Analyzed:**
  * 📌 NIFTY (Index)
  * 📌 TCS
  * 📌 Reliance
  * 📌 Infosys
* **Features Used:**
  * OHLC (Open, High, Low, Close)
  * Volume
  * Technical Indicators: Moving Average (MA5, MA10)

## Models Implemented

### 🔹 Baseline Models
* LSTM
* BiLSTM

### 🔹 Deep Learning Models
* Transformer (Encoder-based)
* Modified Transformer (MTRAN)

### 🔹 Hybrid Models
* CNN + BiLSTM
* CNN + BiLSTM + Attention
* MTRAN + TCN
* BiLSTM + MTRAN + TCN (Final Model) 

## Project Structure
```
stock-price-prediction/
│
├── data/
├── models/
├── utils/
├── train.py
├── config.py
└── README.md
```

## How to Run

1️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

2️⃣ **Train the model**
```bash
python train.py --dataset TCS --model HYBRID
```

**Example:**
```bash
python train.py --dataset NIFTY --model LSTM
```

## Results

| Model | Performance (R²) |
| :--- | :--- |
| LSTM | ~0.95 |
| BiLSTM | ~0.95 |
| CNN-BiLSTM | ~0.87 |
| MTRAN | ~0.83 |
| MTRAN + TCN | ~0.85 |
| **Hybrid (Final Model)** | **~0.95 - 0.98** |

## Key Techniques Used
* Time-series windowing
* Z-score normalization
* Positional Encoding
* Self-Attention mechanism
* Residual connections (improves gradient flow)
* Temporal Convolution (TCN)

## Key Insights
* Hybrid models significantly outperform individual, standalone architectures.
* Residual connections drastically improved model performance and training stability.
* The Transformer effectively captures long-range dependencies in the data.
* Combining sequential and convolutional models yields the best overall predictive results.

## Technologies Used
* **Python**
* **PyTorch**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-learn** (Preprocessing & Metrics)
* **Matplotlib** (Visualization)
* **yFinance** (Data Extraction)

## Conclusion
The proposed hybrid architecture (**BiLSTM + MTRAN + TCN**) achieved the highest predictive performance by successfully capturing:
1. **Short-term patterns** (via TCN)
2. **Sequential dependencies** (via BiLSTM)
3. **Long-range relationships** (via Transformer)

---

### Author
**Arunkumar D** 
```
