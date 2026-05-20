# StockSight — Stock Price Analysis Dashboard

A full-stack web dashboard for stock market analysis built with Dash and Python. It combines deep learning (LSTM), statistical forecasting (ARIMA), and real-time market data to give you a comprehensive view of stock performance — all in one interactive interface.

---

## Live Demo Preview

The app runs as a multi-tab dashboard:

| Tab | Description |
| NSE-TATAGLOBAL — Model Comparison | Compare LSTM, ARIMA, and Moving Average predictions against actual closing prices |
| Live Stock Explorer | Fetch real-time price, volume, and key stats for 30+ global stocks and ETFs |

---

## Features

### Tab 1 — Model Comparison (NSE Tata Global Beverages)
- Trains and compares three forecasting models side by side on historical NSE stock data:
  -  LSTM (Long Short-Term Memory neural network) — deep learning based sequence model
  -  ARIMA (5,1,0) — classical statistical time series model
  - Moving Average (20-day window) — simple baseline model
- Interactive Plotly line chart overlaying all three predictions against actual closing prices
- Performance metrics table showing RMSE and MAE for each model so you can objectively compare accuracy
- Training data vs validation data split clearly visualized

### Tab 2 — Live Stock Explorer
- Dropdown to select from 30+ popular stocks and ETFs including:
  - US Tech: Apple, Microsoft, Google, Amazon, Meta, Tesla, NVIDIA, Netflix, AMD, Intel
  - US Finance: JPMorgan, Goldman Sachs, Visa, Mastercard, Bank of America
  - Indian Stocks (NSE): Reliance, TCS, Infosys, HDFC Bank, Tata Motors, SBI, Wipro, Adani, and more
  - ETFs: SPY (S&P 500), QQQ (NASDAQ), GLD (Gold)
- Flexible time period selector: 1 Month, 3 Months, 6 Months, 1 Year, 2 Years, 5 Years
- Key stats cards displayed at a glance:
  - Current Price, Day Change (%), 52-Week High, 52-Week Low, Market Cap
- Price history chart with Close, High, and Low lines + range slider for zooming
- Trading volume bar chart to spot high-activity periods
- Built-in range selector buttons (1M, 3M, 6M, All) on the price chart

---

## Project Structure

```
├── stock_app_final.py                        # Main Dash app (dashboard + callbacks)
├── stock_pred_model.py                       # Standalone LSTM training script
├── NSE-Tata-Global-Beverages-Limited.csv     # Historical NSE stock dataset
├── saved_lstm_model.h5                       # Pre-trained LSTM model (generated)
├── prediction_plot.png                       # Static prediction chart (generated)
└── favicon_io (2)/                               # App favicon assets
```

---

## Tech Stack

| Category | Library / Tool | Purpose |

| Web Framework | Dash (Plotly) | Interactive multi-tab dashboard UI |
| Data Visualization | Plotly Graph Objects | Line charts, bar charts, metrics table |
| Live Market Data | yfinance | Fetches real-time stock prices and metadata |
| Deep Learning | Keras / TensorFlow | LSTM model for sequence prediction |
| Statistical Modeling | statsmodels (ARIMA) | Classical time series forecasting |
| Data Processing | Pandas, NumPy | Data manipulation and numerical ops |
| Preprocessing | scikit-learn (MinMaxScaler) | Feature normalization for LSTM |
| Metrics | scikit-learn (MSE), math | RMSE and MAE computation |
| Language | Python 3.x | Core language |

---

## Installation

1. Clone the repository
```
git clone https://github.com/Himish04/stocksight.git
cd stocksight
```

2. Install dependencies
```
pip install dash plotly pandas numpy keras tensorflow scikit-learn statsmodels yfinance
```

3. Add the dataset

Place `NSE-Tata-Global-Beverages-Limited.csv` in the root folder. It must contain:
- `Date` — format: `YYYY-MM-DD`
- `Close` — daily closing price

---

## Usage

Step 1 — Train the LSTM model (only needed once)
```bash
python stock_pred_model.py
```
This generates `saved_lstm_model.h5` and `prediction_plot.png`.

Step 2 — Launch the dashboard
```bash
python stock_app_fixed.py
```
Then open your browser at: http://127.0.0.1:8050

---

## LSTM Model Architecture

```
Input (60-day sequences)
    → LSTM (50 units, return_sequences=True)
    → LSTM (50 units)
    → Dense (1 unit)
    → Predicted Closing Price
```

- Loss: Mean Squared Error
- Optimizer: Adam
- Sequence length: 60 days
- Train / Validation split: 987 rows training, remaining rows validation
- Scaler: MinMaxScaler (range 0–1)

---

## Model Comparison — How It Works

All three models are evaluated on the same validation set (data after index 987):

| Model | Approach | Strengths |

| LSTM | Deep learning on 60-day sequences | Captures non-linear long-term patterns |
| ARIMA (5,1,0) | Autoregressive statistical model | Works well on stationary, linear trends |
| Moving Average (20d) | Rolling window average | Simple, fast, interpretable baseline |

Performance is measured using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) — both displayed in the dashboard's metrics table.

---

## Bugs Fixed

| # | Bug | Fix Applied |
|---|---|---|
| 1 | `%matplotlib inline` SyntaxError in `.py` script | Replaced with `matplotlib.use('Agg')` |
| 2 | Unused `Dropout` import | Removed to clean up imports |
| 3 | Chained indexing `df[col][i]` caused silent failures | Replaced with `.loc[]` |
| 4 | `model.predict()` called on undefined variable | Fixed to `lstm_model.predict()` |
| 5 | `prediction_closing` variable undefined | Corrected to `closing_price` with `.flatten()` |

---

## Dataset

The model comparison tab uses NSE Tata Global Beverages Limited historical data. You can source similar datasets from:
- [Kaggle](https://www.kaggle.com/)
- [NSE India](https://www.nseindia.com/)
- [Yahoo Finance](https://finance.yahoo.com/)

The Live Stock tab fetches data directly from Yahoo Finance via the `yfinance` library — no manual downloads needed.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

Himish Goel 
[GitHub](https://github.com/Himish04) • [LinkedIn](https://www.linkedin.com/in/himish-goel-9895b5353/)
