import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")


app = dash.Dash(__name__, suppress_callback_exceptions=True, title = "StockSight", assets_folder='favicon_io')
server = app.server

# ─── Load & prepare NSE-TATA data ─────────────────────────────────────────────

scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data.loc[i, "Date"] = data['Date'].iloc[i]
    new_data.loc[i, "Close"] = data["Close"].iloc[i]

new_data.index = new_data.Date
new_data.index = new_data.index.infer_objects()
new_data.drop("Date", axis=1, inplace=True)
new_data["Close"] = pd.to_numeric(new_data["Close"])

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model("saved_lstm_model.h5")

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train_df = new_data[:987].copy()
valid_df = new_data[987:].copy()
valid_df.loc[:, 'Predictions_LSTM'] = closing_price.flatten()

# ─── ARIMA predictions ────────────────────────────────────────────────────────

print("Fitting ARIMA model...")
arima_model = ARIMA(train_df["Close"].astype(float), order=(5, 1, 0))
arima_fitted = arima_model.fit()
arima_forecast = arima_fitted.forecast(steps=len(valid_df))
valid_df.loc[:, 'Predictions_ARIMA'] = arima_forecast.values

# ─── Moving Average predictions ───────────────────────────────────────────────

window = 20
all_close = new_data["Close"].astype(float).values
ma_preds = []
for i in range(len(train_df), len(new_data)):
    window_vals = all_close[i - window:i]
    ma_preds.append(np.mean(window_vals))
valid_df.loc[:, 'Predictions_MA'] = ma_preds

# ─── Metrics ──────────────────────────────────────────────────────────────────

actual = valid_df["Close"].astype(float).values

def rmse(actual, predicted):
    return round(math.sqrt(mean_squared_error(actual, predicted)), 4)

def mae(actual, predicted):
    return round(np.mean(np.abs(actual - predicted)), 4)

lstm_rmse  = rmse(actual, valid_df['Predictions_LSTM'].values)
arima_rmse = rmse(actual, valid_df['Predictions_ARIMA'].values)
ma_rmse    = rmse(actual, valid_df['Predictions_MA'].values)

lstm_mae   = mae(actual, valid_df['Predictions_LSTM'].values)
arima_mae  = mae(actual, valid_df['Predictions_ARIMA'].values)
ma_mae     = mae(actual, valid_df['Predictions_MA'].values)

print(f"LSTM  — RMSE: {lstm_rmse}  MAE: {lstm_mae}")
print(f"ARIMA — RMSE: {arima_rmse}  MAE: {arima_mae}")
print(f"MA    — RMSE: {ma_rmse}  MAE: {ma_mae}")

# ─── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard",
            style={"textAlign": "center", "fontFamily": "Arial", "paddingTop": "20px"}),

    dcc.Tabs(id="tabs", children=[

        # ── Tab 1: Model Comparison ──────────────────────────────────────────
        dcc.Tab(label='NSE-TATAGLOBAL — Model Comparison', children=[
            html.Div([

                html.H2("LSTM vs ARIMA vs Moving Average",
                        style={"textAlign": "center", "fontFamily": "Arial", "paddingTop": "15px"}),

                dcc.Graph(
                    id="comparison-chart",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_df.index,
                                y=train_df["Close"].astype(float),
                                mode='lines',
                                name='Training Data',
                                line=dict(color='#2196F3')
                            ),
                            go.Scatter(
                                x=valid_df.index,
                                y=valid_df["Close"].astype(float),
                                mode='lines',
                                name='Actual Price',
                                line=dict(color='#4CAF50')
                            ),
                            go.Scatter(
                                x=valid_df.index,
                                y=valid_df['Predictions_LSTM'],
                                mode='lines',
                                name='LSTM Prediction',
                                line=dict(color='#FF5722', dash='dot')
                            ),
                            go.Scatter(
                                x=valid_df.index,
                                y=valid_df['Predictions_ARIMA'],
                                mode='lines',
                                name='ARIMA Prediction',
                                line=dict(color='#9C27B0', dash='dash')
                            ),
                            go.Scatter(
                                x=valid_df.index,
                                y=valid_df['Predictions_MA'],
                                mode='lines',
                                name=f'Moving Average ({window}d)',
                                line=dict(color='#FF9800', dash='dashdot')
                            ),
                        ],
                        "layout": go.Layout(
                            title='Model Predictions vs Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Price (INR)'},
                            legend=dict(x=0, y=1),
                            hovermode='x unified',
                            height=500
                        )
                    }
                ),

                html.H2("Model Performance Metrics",
                        style={"textAlign": "center", "fontFamily": "Arial", "paddingTop": "10px"}),

                dcc.Graph(
                    id="metrics-table",
                    figure={
                        "data": [
                            go.Table(
                                header=dict(
                                    values=["Model", "RMSE", "MAE"],
                                    fill_color='#2196F3',
                                    font=dict(color='white', size=14),
                                    align='center',
                                    height=40
                                ),
                                cells=dict(
                                    values=[
                                        ["LSTM", "ARIMA", "Moving Average (20d)"],
                                        [lstm_rmse, arima_rmse, ma_rmse],
                                        [lstm_mae, arima_mae, ma_mae]
                                    ],
                                    fill_color=[['#f5f5f5', 'white', '#f5f5f5']],
                                    align='center',
                                    font=dict(size=13),
                                    height=35
                                )
                            )
                        ],
                        "layout": go.Layout(
                            height=200,
                            margin=dict(t=10, b=10)
                        )
                    }
                ),

            ], style={"maxWidth": "1100px", "margin": "0 auto"})
        ]),

        # ── Tab 2: Live Stock Data ───────────────────────────────────────────
        dcc.Tab(label='Live Stock Data', children=[
            html.Div([

                html.H2("Live Stock Explorer",
                        style={"textAlign": "center", "fontFamily": "Arial", "paddingTop": "15px"}),

                html.Div([
                    dcc.Dropdown(
                        id='ticker-input',
                        options=[
                            # US Tech
                            {'label': 'Apple (AAPL)',               'value': 'AAPL'},
                            {'label': 'Microsoft (MSFT)',           'value': 'MSFT'},
                            {'label': 'Google (GOOGL)',             'value': 'GOOGL'},
                            {'label': 'Amazon (AMZN)',              'value': 'AMZN'},
                            {'label': 'Meta (META)',                'value': 'META'},
                            {'label': 'Tesla (TSLA)',               'value': 'TSLA'},
                            {'label': 'Netflix (NFLX)',             'value': 'NFLX'},
                            {'label': 'NVIDIA (NVDA)',              'value': 'NVDA'},
                            {'label': 'AMD (AMD)',                  'value': 'AMD'},
                            {'label': 'Intel (INTC)',               'value': 'INTC'},
                            # US Finance
                            {'label': 'JPMorgan Chase (JPM)',       'value': 'JPM'},
                            {'label': 'Goldman Sachs (GS)',         'value': 'GS'},
                            {'label': 'Bank of America (BAC)',      'value': 'BAC'},
                            {'label': 'Visa (V)',                   'value': 'V'},
                            {'label': 'Mastercard (MA)',            'value': 'MA'},
                            # US Others
                            {'label': 'Johnson & Johnson (JNJ)',    'value': 'JNJ'},
                            {'label': 'Walmart (WMT)',              'value': 'WMT'},
                            {'label': 'Coca-Cola (KO)',             'value': 'KO'},
                            {'label': 'ExxonMobil (XOM)',           'value': 'XOM'},
                            {'label': 'Berkshire Hathaway (BRK-B)','value': 'BRK-B'},
                            # Indian Stocks (NSE)
                            {'label': 'Tata Motors (NSE)',         'value': 'TATAMOTORS.NS'},
                            {'label': 'Reliance Industries (NSE)', 'value': 'RELIANCE.NS'},
                            {'label': 'Infosys (NSE)',             'value': 'INFY.NS'},
                            {'label': 'TCS (NSE)',                 'value': 'TCS.NS'},
                            {'label': 'HDFC Bank (NSE)',           'value': 'HDFCBANK.NS'},
                            {'label': 'ICICI Bank (NSE)',          'value': 'ICICIBANK.NS'},
                            {'label': 'Wipro (NSE)',               'value': 'WIPRO.NS'},
                            {'label': 'Bajaj Finance (NSE)',       'value': 'BAJFINANCE.NS'},
                            {'label': 'Adani Enterprises (NSE)',   'value': 'ADANIENT.NS'},
                            {'label': 'SBI (NSE)',                 'value': 'SBIN.NS'},
                            # ETFs
                            {'label': 'S&P 500 ETF (SPY)',         'value': 'SPY'},
                            {'label': 'NASDAQ ETF (QQQ)',          'value': 'QQQ'},
                            {'label': 'Gold ETF (GLD)',            'value': 'GLD'},
                        ],
                        value='AAPL',
                        clearable=False,
                        searchable=True,
                        placeholder='Search or select a stock...',
                        style={"width": "380px", "marginRight": "10px",
                               "fontSize": "15px"}
                    ),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Month',  'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year',   'value': '1y'},
                            {'label': '2 Years',  'value': '2y'},
                            {'label': '5 Years',  'value': '5y'},
                        ],
                        value='1y',
                        clearable=False,
                        style={"width": "150px", "marginRight": "10px",
                               "fontSize": "15px"}
                    ),
                    html.Button(
                        'Fetch',
                        id='fetch-button',
                        n_clicks=0,
                        style={
                            "padding": "10px 20px", "fontSize": "15px",
                            "backgroundColor": "#2196F3", "color": "white",
                            "border": "none", "borderRadius": "5px", "cursor": "pointer"
                        }
                    )
                ], style={"display": "flex", "justifyContent": "center",
                          "alignItems": "center", "marginBottom": "20px", "flexWrap": "wrap"}),

                # Key stats cards
                html.Div(id='stats-row', style={
                    "display": "flex", "justifyContent": "center",
                    "gap": "20px", "marginBottom": "20px", "flexWrap": "wrap"
                }),

                dcc.Graph(id='live-price-chart'),
                dcc.Graph(id='live-volume-chart'),

            ], style={"maxWidth": "1100px", "margin": "0 auto"})
        ]),

    ])
], style={"fontFamily": "Arial"})


# ─── Callback: Live Stock Tab ──────────────────────────────────────────────────

@app.callback(
    Output('live-price-chart', 'figure'),
    Output('live-volume-chart', 'figure'),
    Output('stats-row', 'children'),
    Input('fetch-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('period-dropdown', 'value')
)
def update_live_charts(n_clicks, ticker, period):
    ticker = (ticker or 'AAPL').upper()

    try:
        df_live = yf.download(ticker, period=period, progress=False)
        info = yf.Ticker(ticker).info
    except Exception:
        empty = go.Figure()
        empty.update_layout(title="Could not fetch data. Check ticker symbol.")
        return empty, empty, []

    if df_live.empty:
        empty = go.Figure()
        empty.update_layout(title=f"No data found for '{ticker}'. Check the ticker symbol.")
        return empty, empty, []

    df_live.reset_index(inplace=True)

    # Price chart
    price_fig = {
        "data": [
            go.Scatter(
                x=df_live["Date"],
                y=df_live["Close"].squeeze(),
                mode='lines',
                name='Close Price',
                line=dict(color='#2196F3')
            ),
            go.Scatter(
                x=df_live["Date"],
                y=df_live["High"].squeeze(),
                mode='lines',
                name='High',
                line=dict(color='#4CAF50', dash='dot'),
                opacity=0.6
            ),
            go.Scatter(
                x=df_live["Date"],
                y=df_live["Low"].squeeze(),
                mode='lines',
                name='Low',
                line=dict(color='#FF5722', dash='dot'),
                opacity=0.6
            ),
        ],
        "layout": go.Layout(
            title=f"{ticker} — Price History ({period})",
            xaxis={
                'title': 'Date',
                'rangeselector': {'buttons': list([
                    {'count': 1,  'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 3,  'label': '3M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6,  'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                    {'step': 'all', 'label': 'All'}
                ])},
                'rangeslider': {'visible': True},
                'type': 'date'
            },
            yaxis={'title': 'Price (USD)'},
            hovermode='x unified',
            height=450,
            legend=dict(x=0, y=1)
        )
    }

    # Volume chart
    volume_fig = {
        "data": [
            go.Bar(
                x=df_live["Date"],
                y=df_live["Volume"].squeeze(),
                name='Volume',
                marker_color='#9C27B0',
                opacity=0.7
            )
        ],
        "layout": go.Layout(
            title=f"{ticker} — Trading Volume ({period})",
            xaxis={'title': 'Date', 'type': 'date'},
            yaxis={'title': 'Volume'},
            height=300
        )
    }

    # Key stats cards
    def stat_card(label, value):
        return html.Div([
            html.P(label, style={"margin": "0", "fontSize": "12px", "color": "#888"}),
            html.P(value, style={"margin": "0", "fontSize": "18px",
                                 "fontWeight": "bold", "color": "#2196F3"})
        ], style={
            "background": "#f9f9f9", "borderRadius": "8px",
            "padding": "15px 25px", "textAlign": "center",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.1)", "minWidth": "140px"
        })

    current_price = df_live["Close"].iloc[-1].squeeze()
    prev_close    = df_live["Close"].iloc[-2].squeeze() if len(df_live) > 1 else current_price
    change_pct    = ((current_price - prev_close) / prev_close) * 100

    market_cap = info.get("marketCap", None)
    market_cap_str = f"${market_cap / 1e9:.2f}B" if market_cap else "N/A"

    week_high = info.get("fiftyTwoWeekHigh", None)
    week_low  = info.get("fiftyTwoWeekLow",  None)

    stats = [
        stat_card("Current Price", f"${current_price:.2f}"),
        stat_card("Day Change",    f"{change_pct:+.2f}%"),
        stat_card("52W High",      f"${week_high:.2f}" if week_high else "N/A"),
        stat_card("52W Low",       f"${week_low:.2f}"  if week_low  else "N/A"),
        stat_card("Market Cap",    market_cap_str),
    ]

    return price_fig, volume_fig, stats


if __name__ == '__main__':
    app.run(debug=False)
