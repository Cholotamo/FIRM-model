import pandas as pd
import numpy as np
import os

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates MACD, its signal line, and histogram.
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculates Bollinger Bands: SMA, Upper Band, and Lower Band.
    """
    sma = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return sma, upper_band, lower_band

def rsi_signal(rsi_value):
    """
    Generates signal for RSI: Buy if oversold, Sell if overbought, Hold otherwise.
    """
    if rsi_value < 30:
        return 'Buy'
    elif rsi_value > 70:
        return 'Sell'
    else:
        return 'Hold'

def macd_signal(macd_line, signal_line):
    """
    Generates signal for MACD: Buy if MACD line is above its signal, Sell if below.
    """
    if macd_line > signal_line:
        return 'Buy'
    elif macd_line < signal_line:
        return 'Sell'
    else:
        return 'Hold'

def bollinger_signal(price, lower_band, upper_band):
    """
    Generates signal for Bollinger Bands: Buy if price below lower band, Sell if above.
    """
    if price < lower_band:
        return 'Buy'
    elif price > upper_band:
        return 'Sell'
    else:
        return 'Hold'

def overall_signal(rsi_sig, macd_sig, bb_sig):
    """
    Combines the three indicator signals into an overall signal using a simple scoring:
      Buy: +1, Sell: -1, Hold: 0.
    """
    mapping = {'Buy': 1, 'Sell': -1, 'Hold': 0}
    total_score = mapping[rsi_sig] + mapping[macd_sig] + mapping[bb_sig]
    
    if total_score > 0:
        return 'Buy'
    elif total_score < 0:
        return 'Sell'
    else:
        return 'Hold'

def main():
    # Define input and output directories
    input_dir = "input_data_cleaning"
    output_dir = "output_data_cleaning"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Define input and output file paths
    input_file = os.path.join(input_dir, "KO_HP.xlsx")
    output_file = os.path.join(output_dir, "KO_HP_with_signals.csv")

    # Read the Excel file, skipping metadata rows so that row 7 is the header.
    df = pd.read_excel(input_file, skiprows=6)
    
    print("Columns read by pandas:", df.columns.tolist())
    
    # Check that required columns exist.
    date_col = 'Date'
    price_col = 'PX_LAST'
    if date_col not in df.columns or price_col not in df.columns:
        raise KeyError(f"Required columns '{date_col}' and/or '{price_col}' not found.\nColumns: {df.columns.tolist()}")
    
    # Convert Date to datetime (using dayfirst=True if dates are in DD/MM/YYYY format)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df.sort_values(by=date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Create a Price column for clarity.
    df['Price'] = df[price_col]
    
    # Calculate technical indicators.
    df['RSI'] = calculate_rsi(df['Price'])
    df['MACD_Line'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Price'])
    df['Bollinger_SMA'], df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Price'])
    
    # Generate individual signals for RSI, MACD, and Bollinger Bands.
    df['RSI_Signal'] = df['RSI'].apply(rsi_signal)
    df['MACD_Signal_Indicator'] = df.apply(lambda x: macd_signal(x['MACD_Line'], x['MACD_Signal']), axis=1)
    df['Bollinger_Signal'] = df.apply(lambda x: bollinger_signal(x['Price'], x['Bollinger_Lower'], x['Bollinger_Upper']), axis=1)
    
    # Overall signal based on RSI, MACD, and Bollinger Bands.
    df['Overall_Signal'] = df.apply(lambda x: overall_signal(x['RSI_Signal'], x['MACD_Signal_Indicator'], x['Bollinger_Signal']), axis=1)
    
    # --- Additional Section: 9-day and 20-day SMA and EMA with Crossover Signals ---
    # Calculate 9-day and 20-day SMA.
    df['SMA_9'] = df['Price'].rolling(window=9, min_periods=9).mean()
    df['SMA_20'] = df['Price'].rolling(window=20, min_periods=20).mean()
    
    # Calculate 9-day and 20-day EMA.
    df['EMA_9'] = df['Price'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Price'].ewm(span=20, adjust=False).mean()
    
    # Determine crossovers for SMA and EMA.
    df['SMA_diff'] = df['SMA_9'] - df['SMA_20']
    df['EMA_diff'] = df['EMA_9'] - df['EMA_20']
    
    df['SMA_cross'] = df['SMA_diff'].apply(np.sign)
    df['EMA_cross'] = df['EMA_diff'].apply(np.sign)
    
    # Initialize crossover signals.
    df['SMA_Cross_Signal'] = 'Hold'
    df['EMA_Cross_Signal'] = 'Hold'
    
    # If the sign of the SMA difference changes from negative to positive, that's a Buy signal; from positive to negative, a Sell.
    df.loc[(df['SMA_cross'].diff() > 0), 'SMA_Cross_Signal'] = 'Buy'
    df.loc[(df['SMA_cross'].diff() < 0), 'SMA_Cross_Signal'] = 'Sell'
    
    # Similarly for EMA.
    df.loc[(df['EMA_cross'].diff() > 0), 'EMA_Cross_Signal'] = 'Buy'
    df.loc[(df['EMA_cross'].diff() < 0), 'EMA_Cross_Signal'] = 'Sell'
    
    # --- Rearranging Columns ---
    # Define an ordered list of columns for a cleaner output.
    ordered_cols = [
        'Date', 'PX_LAST', 'PX_BID', 'Price',
        'RSI', 'RSI_Signal',
        'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'MACD_Signal_Indicator',
        'Bollinger_SMA', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Signal',
        'SMA_9', 'SMA_20', 'EMA_9', 'EMA_20', 'SMA_Cross_Signal', 'EMA_Cross_Signal',
        'Overall_Signal'
    ]
    
    # Rearrange the dataframe columns.
    df = df[ordered_cols]
    
    # Export the final DataFrame to a new CSV file.
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results exported to: {output_file}")

if __name__ == "__main__":
    main()
