import os
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from ta.trend import MACD
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import telebot
from telebot.types import Message

# Initialize the Telegram bot
bot = telebot.TeleBot('YOUR_TOKEN_HERE')
print('Bot started')

# Function to get the crypto symbol from the user
def get_crypto_symbol(message: Message):
    crypto_symbol = message.text.upper()
    return crypto_symbol

# Import Bitcoin Price Data from Binance
def get_binance_data(crypto_symbol):
    print(f"Getting {crypto_symbol} data from Binance...")
    binance = ccxt.binance()
    symbol = crypto_symbol
    timeframe = '1d'
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=1000)

    if ohlcv:
        df = pd.DataFrame(ohlcv, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        df = df.apply(pd.to_numeric)
        return df
    else:
        print('No data found')
        return None

# Save the data to a CSV file
def save_data(df, crypto_symbol):
    df.to_csv(f'{crypto_symbol}.csv')
    print(f"Data saved to {crypto_symbol}.csv")
    os.remove(f'{crypto_symbol}.csv')

# Load the data from a CSV file
def load_data(crypto_symbol):
    df = pd.read_csv(f'{crypto_symbol}.csv', index_col='Open Time', parse_dates=True)
    return df

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Clean the data
def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['MA'] = df['Close'].rolling(window=20).mean()
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    return df

# Train the model with the data
def train_model(df):
    X = df.drop('Close', axis=1)
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    column_transformer = make_column_transformer(
        (SimpleImputer(), X_train.columns),
        remainder='passthrough'
    )

    models = [
        ('Random Forest', make_pipeline(column_transformer, RandomForestRegressor())),
        ('Decision Tree', make_pipeline(column_transformer, DecisionTreeRegressor())),
        ('Linear Regression', make_pipeline(column_transformer, LinearRegression())),
        ('Ridge', make_pipeline(column_transformer, Ridge()))
    ]

    model_scores = {}
    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        model_scores[name] = score
    return model_scores

# Print the model scores
def print_model_scores(model_scores):
    for i, (name, score) in enumerate(model_scores.items()):
        model_name = list(model_scores.keys())[i]
        print(f'{model_name} Model R-squared: {score*100:.2f}%')

# Predict the price for the next 30 days
def predict_price(df):
    X = df.drop('Close', axis=1)
    y = df['Close']

    imputer = SimpleImputer()
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    models = [
        ('Random Forest', RandomForestRegressor()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge())
    ]

    predictions = []
    for name, model in models:
        model.fit(X, y)
        prediction = model.predict(X[-30:])
        predictions.append(prediction)

    predictions = np.array(predictions).T
    return predictions

# Save the predictions to a symbol name.csv file
def save_predictions(predictions, crypto_symbol, file_name):
    df = pd.DataFrame(predictions)
    model_names = [f'Model {i+1}' for i in range(predictions.shape[1])]
    df.columns = model_names
    df.to_csv(file_name)
    print(f"Predictions saved to {file_name}")

# Advanced interactive plot with Plotly (Dropdown, Range Slider, Annotations)
def plot_predictions_advanced(crypto_symbol, file_name):
    predictions_df = pd.read_csv(file_name)
    
    # Create a figure
    fig = go.Figure()

    # Add traces for each model's predictions
    for column in predictions_df.columns[1:]:
        fig.add_trace(go.Scatter(
            x=predictions_df.index,
            y=predictions_df[column],
            mode='lines+markers',
            name=column,
            visible=True if column == 'Model 1' else False  # Set only Model 1 visible initially
        ))

    # Dropdown menu for selecting models to display
    dropdown_buttons = []
    for column in predictions_df.columns[1:]:
        dropdown_buttons.append(dict(
            args=[{'visible': [col == column for col in predictions_df.columns[1:]]}],
            label=column,
            method='restyle'
        ))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.15,  # Place it to the right
                y=0.5
            )
        ]
    )

    # Range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )
    )

    # Add title and labels
    fig.update_layout(
        title=f'Predictions for {crypto_symbol} - Model Comparison',
        xaxis_title='Days',
        yaxis_title='Predicted Price',
        template='plotly_white',
        hovermode='x unified',
        legend_title_text='Models',
    )

    # Show the interactive plot in the browser or supported terminal
    fig.show()

# Handle the '/start' command
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "Welcome to the Crypto Price Prediction bot!")

# Handle the '/predict' command
@bot.message_handler(commands=['predict'])
def handle_predict(message):
    bot.reply_to(message, "Enter the symbol of the cryptocurrency (e.g., BTCUSDT):")
    bot.register_next_step_handler(message, process_crypto_symbol)

# Process the cryptocurrency symbol
def process_crypto_symbol(message):
    crypto_symbol = message.text
    data = get_binance_data(crypto_symbol)
    if data is not None:
        cleaned_data = clean_data(data)
        model_scores = train_model(cleaned_data)
        print_model_scores(model_scores)
        predictions = predict_price(cleaned_data)
        
        # Save CSV in both Telegram and root folder
        file_name = f'{crypto_symbol}_predictions.csv'
        save_predictions(predictions, crypto_symbol, file_name)
        
        # Send the CSV to Telegram chat
        bot.send_document(message.chat.id, open(file_name, 'rb'))

        # Plot predictions interactively
        plot_predictions_advanced(crypto_symbol, file_name)
    else:
        bot.reply_to(message, "No data found for the given cryptocurrency symbol.")

# Handle all other messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.reply_to(message, "Invalid command. Please use /start to begin.")

# Start the bot
bot.polling()
