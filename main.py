import os
import ccxt
import numpy as np
import pandas as pd
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
bot = telebot.TeleBot('5965168157:AAHzgqx8nOhcBFGtwTK6ZqUbZfDkIoq9BlM')
print('Bot started')

# Function to get the crypto symbol from the user
def get_crypto_symbol(message: Message):
    crypto_symbol = message.text.upper()
    return crypto_symbol

# Import Bitcoin Price Data from Binance
def get_binance_data(crypto_symbol):
    print(f"Getting {crypto_symbol} data from Binance...")
    # Create a Binance exchange instance
    binance = ccxt.binance()

    # Define the symbol and timeframe
    symbol = crypto_symbol
    timeframe = '1d'

    # Fetch OHLCV (Open-High-Low-Close-Volume) data
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=1000)

    # if data is not empty
    if ohlcv:
        # Create a DataFrame
        df = pd.DataFrame(ohlcv, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert the timestamp to datetime
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')

        # Set the index to the date column
        df.set_index('Open Time', inplace=True)

        # Convert the columns to numeric
        df = df.apply(pd.to_numeric)

        # Return the DataFrame
        return df
    else:
        print('No data found')
        return None

# Save the data to a CSV file
def save_data(df, crypto_symbol):
    # Save the data to a CSV file
    df.to_csv(f'{crypto_symbol}.csv')
    print(f"Data saved to {crypto_symbol}.csv")

    os.remove(f'{crypto_symbol}.csv')

# Load the data from a CSV file
def load_data(crypto_symbol):
    # Load the data from a CSV file
    df = pd.read_csv(f'{crypto_symbol}.csv', index_col='Open Time', parse_dates=True)
    # Return the DataFrame
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
    # Drop the rows with missing values
    df.dropna(inplace=True)
    # Drop the duplicate rows
    df.drop_duplicates(inplace=True)

    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], window=14)

    # Calculate Moving Average
    df['MA'] = df['Close'].rolling(window=20).mean()

    # Calculate MACD
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()

    # Return the cleaned DataFrame
    return df

# Train the model with the data
def train_model(df):
    # Create the features and target
    X = df.drop('Close', axis=1)
    y = df['Close']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the column transformer
    column_transformer = make_column_transformer(
        (SimpleImputer(), X_train.columns),
        remainder='passthrough'
    )

    # Create a list of models
    models = [
        ('Random Forest', make_pipeline(column_transformer, RandomForestRegressor())),
        ('Decision Tree', make_pipeline(column_transformer, DecisionTreeRegressor())),
        ('Linear Regression', make_pipeline(column_transformer, LinearRegression())),
        ('Ridge', make_pipeline(column_transformer, Ridge()))
    ]

    # Create a dictionary to store the model scores
    model_scores = {}

    # Loop through the models
    for name, model in models:
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and store its score
        score = model.score(X_test, y_test)
        model_scores[name] = score

    # Return the model scores
    return model_scores


# Print the model scores
def print_model_scores(model_scores):
    # Loop through the model names and scores and print them
    for i, (name, score) in enumerate(model_scores.items()):
        model_name = list(model_scores.keys())[i]  # Extract the model name from the keys
        print(f'{model_name} Model R-squared: {score*100:.2f}%')

# Predict the price for the next 30 days
def predict_price(df):
    # Create the features and target
    X = df.drop('Close', axis=1)
    y = df['Close']

    # Impute missing values
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Create a list of models
    models = [
        ('Random Forest', RandomForestRegressor()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge())
    ]
    # Create a list to store the predictions
    predictions = []
    # Loop through the models
    for name, model in models:
        # Fit the model to the data
        model.fit(X, y)
        # Predict the price for the next 30 days
        prediction = model.predict(X[-30:])
        # Append the predictions to the list
        predictions.append(prediction)
    # Convert the predictions list to a numpy array
    predictions = np.array(predictions).T
    # Return the predictions
    return predictions

# Save the predictions to a symbol name.csv file
def save_predictions(predictions, crypto_symbol, file_name):
    # Create a DataFrame
    df = pd.DataFrame(predictions)
    # Get the model names based on the number of columns
    model_names = [f'Model {i+1}' for i in range(predictions.shape[1])]
    # Set the columns of the DataFrame
    df.columns = model_names
    # Save the predictions to a CSV file
    df.to_csv(file_name)
    print(f"Predictions saved to {file_name}")

# Handle the '/start' command
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "Welcome to the Crypto Price Prediction bot!")

# Handle the '/predict' command
@bot.message_handler(commands=['predict'])
def handle_predict(message):
    # Ask the user for the symbol of the cryptocurrency
    bot.reply_to(message, "Enter the symbol of the cryptocurrency (e.g., BTC):")
    # Register the next handler to receive the symbol from the user
    bot.register_next_step_handler(message, process_crypto_symbol)


# Process the cryptocurrency symbol
def process_crypto_symbol(message):
    crypto_symbol = message.text
    # Get the data from Binance
    data = get_binance_data(crypto_symbol)
    if data is not None:
        # Clean the data
        cleaned_data = clean_data(data)
        # Train the model
        model_scores = train_model(cleaned_data)
        # Print the model scores
        print_model_scores(model_scores)
        # Predict the price for the next 30 days
        predictions = predict_price(cleaned_data)
        # Save the predictions to a CSV file
        file_name = f'{crypto_symbol}_predictions.csv'
        save_predictions(predictions, crypto_symbol, file_name)
        # Send the predictions file to the user
        bot.send_document(message.chat.id, open(file_name, 'rb'))
        # Delete the CSV file
        os.remove(file_name)
    else:
        bot.reply_to(message, "No data found for the given cryptocurrency symbol.")


# Handle all other messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.reply_to(message, "Invalid command. Please use /start to begin.")

# Start the bot
bot.polling()
