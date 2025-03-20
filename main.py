#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
import pandas as pd
from pyperclip import copy as pyperclip_copy
from datetime import datetime, timezone
from binance.um_futures import UMFutures
from binance.error import ClientError, ServerError

def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.
    
    Args:
        prices: List of closing prices
        period: RSI period (default: 14)
        
    Returns:
        List of RSI values (with NaN for the first period-1 values)
    """
    # Convert to numpy array for calculations
    prices_array = np.array(prices)
    # Calculate price changes
    deltas = np.diff(prices_array)
    # Create arrays of gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initialize average gains and losses
    avg_gain = np.concatenate(([np.nan] * (period - 1), [np.mean(gains[:period])]))
    avg_loss = np.concatenate(([np.nan] * (period - 1), [np.mean(losses[:period])]))
    
    # Calculate subsequent values
    for i in range(period, len(prices_array)):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * (period - 1) + gains[i-1]) / period)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * (period - 1) + losses[i-1]) / period)
    
    # Calculate RS and RSI
    rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.tolist()

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) for a series of prices.
    
    Args:
        prices: List of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal EMA period (default: 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Convert to pandas Series for easy EMA calculation
    close_series = pd.Series(prices)
    
    # Calculate EMAs
    ema_fast = close_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line.tolist(), signal_line.tolist(), histogram.tolist()

def calculate_sma(prices, period=20):
    """
    Calculate Simple Moving Average (SMA) for a series of prices.
    
    Args:
        prices: List of closing prices
        period: SMA period (default: 20)
        
    Returns:
        List of SMA values (with NaN for the first period-1 values)
    """
    # Create a pandas Series for the calculation
    close_series = pd.Series(prices)
    
    # Calculate SMA
    sma = close_series.rolling(window=period).mean()
    
    return sma.tolist()

def fetch_historical_data_with_indicators(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    filename: str | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26, 
    macd_signal: int = 9,
    sma_period: int = 20
) -> str | None:
    """
    Fetch historical klines data from Binance Futures, calculate indicators, and save to CSV.

    Args:
        symbol: Trading pair (e.g., 'SOLUSDT')
        interval: Kline interval (e.g., '1h', '4h', '1d')
        start_time: Start time in epoch milliseconds
        end_time: End time in epoch milliseconds
        filename: Output CSV filename. If None, generate a default name.
        rsi_period: Period for RSI calculation
        macd_fast: Fast period for MACD calculation
        macd_slow: Slow period for MACD calculation
        macd_signal: Signal period for MACD calculation
        sma_period: Period for SMA calculation

    Returns:
        Filename if successful, None otherwise
    """
    client = UMFutures()

    if start_time >= end_time:
        print("Error: Start time must be before end time.")
        return None
    
    # We need to fetch additional historical data to properly calculate indicators
    # Determine the maximum lookback period needed
    max_lookback = max(rsi_period, macd_slow, sma_period)
    
    # Convert interval to minutes for additional time calculation
    interval_minutes = 1  # Default for '1m'
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60
    elif interval.endswith('d'):
        interval_minutes = int(interval[:-1]) * 60 * 24
    elif interval.endswith('w'):
        interval_minutes = int(interval[:-1]) * 60 * 24 * 7
    elif interval.endswith('M'):
        interval_minutes = int(interval[:-1]) * 60 * 24 * 7 * 30
    
    # Adjust start time to include lookback data
    lookback_start_time = start_time - (max_lookback * interval_minutes * 60 * 1000)
    
    all_klines = []
    current_start = lookback_start_time

    try:
        while current_start < end_time:
            klines = client.klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_time,
                limit=1000
            )

            if not klines:
                break

            all_klines.extend(klines)
            last_ts = klines[-1][0]

            if last_ts <= current_start:
                break

            current_start = last_ts + 1
            time.sleep(0.1)

        if not all_klines:
            print("No data retrieved from Binance API.")
            return None
            
        # Convert klines to DataFrame for easier manipulation
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert string values to appropriate types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate indicators
        df['rsi_14'] = calculate_rsi(df['close'].values, rsi_period)
        
        macd_line, signal_line, histogram = calculate_macd(
            df['close'].values, macd_fast, macd_slow, macd_signal
        )
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        df['sma_20'] = calculate_sma(df['close'].values, sma_period)
        
        # Convert timestamp to datetime for better readability
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove lookback period data from final output and keep only requested timeframe
        df_final = df[df['timestamp'] >= start_time].copy()
        
        # Generate filename if not provided
        if filename is None:
            start_dt = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)
            start_str = start_dt.strftime("%d:%H:%M")
            end_str = end_dt.strftime("%d:%H:%M")
            filename = f"{symbol}_{interval}_{start_str}-{end_str}.csv".replace(':', '')

        # Write to CSV, including only the columns we need
        columns_to_save = [
            'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'sma_20'
        ]
        
        csv_data = df_final[columns_to_save].to_csv(index=False)
        
        # Write CSV data to file
        with open(filename, 'w') as f:
            f.write(csv_data)

        print(f"Successfully saved {len(df_final)} records to {filename}")
        print("Technical indicators included: RSI-14, MACD-12/26/9, SMA-20")
        return filename, csv_data

    except ClientError as e:
        print(f"Binance API error: {e.error_code} - {e.error_message}")
    except ServerError as e:
        print(f"Binance server error: {e.status_code}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    return None

def parse_time(input_time: str | None) -> int:
    if input_time is None:
        return int(time.time() * 1000)
    try:
        ts = int(input_time)
        return ts * 1000 if ts < 1e12 else ts
    except ValueError:
        formats = [
            '%Y-%m-%d %H:%M', '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(input_time, fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        raise ValueError(f"Invalid datetime format: {input_time}")

def parse_interval_to_minutes(interval: str) -> int:
    """
    Convert interval string to minutes.
    
    Args:
        interval: Interval string (e.g., '1m', '4h', '1d')
        
    Returns:
        Number of minutes in the interval
    """
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 60 * 24
    elif interval.endswith('w'):
        return int(interval[:-1]) * 60 * 24 * 7
    else:
        return 1  # Default to 1 minute if unknown format

def copy(*args: str):
    text = " ".join(args)

    pyperclip_copy(text)

def print_llm_prompt(symbol: str, start_time: int, end_time: int, interval: str, filename: str, csv_data: str):
    # Define the various prompts in a dictionary for clarity
    prompts = {
        "1": f"""Here is the data for the {symbol} Binance Futures pair. The data spans from {start_time} to {end_time} with each data point representing a {interval} period.
Please analyze the data in depth, using your full reasoning capabilities without constraints. Identify overall market trends, key support and resistance levels, and evaluate both long and short trading opportunities. Aim for strategies that target a profit range of approximately 30%-50% while maintaining sound risk management. Also, recommend appropriate leverage to safely achieve these goals.
For clarity, structure your response as follows:
Brief Overview:
    Summarize the current market conditions, including general trends, notable supports, and resistances.
Short Position Entry:
    Entry Price:
    Exit Price:
    Stop Loss:
    Estimated Time Frame: (How long you expect the trade to play out)
    Relative Probability: (Confidence level of the trade)
    Entry and Exit Conditions:
Long Position Entry:
    Entry Price:
    Exit Price:
    Stop Loss:
    Estimated Time Frame: (How long you expect the trade to play out)
    Relative Probability: (Confidence level of the trade)
    Entry and Exit Conditions:
If you assess that market conditions are too noisy or the timing isn’t right, please highlight this clearly and advise accordingly.
Feel free to include any additional analysis or insights that might not strictly fit the above format if they could further improve the trade strategy.
Thank you.
""",
        "2": f"""Here is the latest update to the data for the {symbol} Binance Futures pair. This new dataset spans from {start_time} to {end_time} with each data point representing a {interval} period. Please integrate this updated information into your previous analysis and revise your trading strategy recommendations accordingly. Update any relevant market trends, support/resistance levels, and entry/exit suggestions based on this fresh data.
Thank you.
""",
        "3": f"""For the {symbol} Binance Futures pair data spanning from {start_time} to {end_time} with each data point representing a {interval} period, please conduct a detailed analysis to identify the key support and resistance zones. Discuss how these critical levels have evolved, indicate potential breakout points, and explain their significance in the current market context. Your insights will help in pinpointing crucial price levels that could drive future trading decisions.
Thank you.
""",
        "4": f"""Analyze the market data for the {symbol} Binance Futures pair covering the period from {start_time} to {end_time} with each data point representing a {interval} period. Based on current trends, technical indicators, and price action, please assess whether a breakout appears imminent. Provide detailed reasoning and reference any key patterns or levels that support your conclusion.
Thank you.
"""
    }

    def get_valid_input(prompt_message: str, valid_choices: list) -> str:
        """Repeatedly prompt the user until a valid option or exit command is entered."""
        while True:
            user_input = input(prompt_message).strip().lower()
            if user_input in ['q', 'exit']:
                print("Exiting.")
                exit()  # Alternatively, you can use "return None" and handle it in the caller.
            if user_input in valid_choices:
                return user_input
            print(f"Invalid input. Please enter one of {valid_choices} or 'q' to exit.")

    # Prompt the user for the type of analysis prompt
    prompt_choice = get_valid_input(
        "Which prompt do you want to use:\n"
        "  1) General Prompt (asks for support, resistance, and entries)\n"
        "  2) Followup Prompt\n"
        "  3) Prompt for asking about support and resistance levels\n"
        "  4) Prompt for asking if a breakout is possible\n"
        "Your choice (or 'q' to exit): ", ["1", "2", "3", "4"]
    )
    prompt_to_use = prompts[prompt_choice]

    # Prompt the user for output option (print or copy)
    copy_choice = get_valid_input(
        "Select one of the following options:\n"
        "  1) Print to console\n"
        "  2) Copy prompt to clipboard\n"
        "  3) Copy prompt + data to clipboard\n"
        "  4) Copy only the data\n"
        "Your choice (or 'q' to exit): ", ["1", "2", "3", "4"]
    )

    # Ensure that the file can be opened
    try:
        with open(f'./{filename}', 'r') as data_file:
            pass
    except Exception as e:
        print("ERROR: COULD NOT OPEN FILE")
        return

    # Execute based on the output choice
    if copy_choice == '1':
        print("\n" + prompt_to_use)
    elif copy_choice == '2':
        copy(prompt_to_use)
        print("Prompt copied to clipboard.")
    elif copy_choice == '3':
        copy(prompt_to_use, csv_data)
        print("Prompt and data copied to clipboard.")
    elif copy_choice == '4':
        copy(csv_data)
        print("Data copied to clipboard.")

    # Ask if the user wants to delete the file and validate the input
    delete_choice = get_valid_input(
        "Do you want to delete the file with the data? (y/n, or 'q' to exit): ", ["y", "n"]
    )
    if delete_choice == 'y':
        if os.path.exists(filename):
            os.remove(filename)
            print("File deleted successfully.")
        else:
            print("File not found.")

def main():
    parser = argparse.ArgumentParser(description='Fetch historical data from Binance Futures with technical indicators.')
    parser.add_argument('-symbol', required=True, help='Trading pair symbol (e.g., SOLUSDT)')
    parser.add_argument('-tframe', required=True, type=int, help='Time frame in number of intervals (how many candles back from end time)')
    parser.add_argument('-interval', default='1m', help='Kline interval (default: 1m)')
    parser.add_argument('-endtime', help='End time (datetime string or epoch timestamp)')
    parser.add_argument('-filename', help='Custom output filename (optional)')
    parser.add_argument('-rsi', type=int, default=14, help='RSI period (default: 14)')
    parser.add_argument('-macd_fast', type=int, default=12, help='MACD fast period (default: 12)')
    parser.add_argument('-macd_slow', type=int, default=26, help='MACD slow period (default: 26)')
    parser.add_argument('-macd_signal', type=int, default=9, help='MACD signal period (default: 9)')
    parser.add_argument('-sma', type=int, default=20, help='SMA period (default: 20)')
    
    args = parser.parse_args()

    try:
        end_time = parse_time(args.endtime) if args.endtime else int(time.time() * 1000)
        # Use the interval in minutes from the provided interval
        interval_minutes = parse_interval_to_minutes(args.interval)
        start_time = end_time - (args.tframe * interval_minutes * 60 * 1000)
        
        result = fetch_historical_data_with_indicators(
            symbol=args.symbol,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time,
            filename=args.filename,
            rsi_period=args.rsi,
            macd_fast=args.macd_fast,
            macd_slow=args.macd_slow,
            macd_signal=args.macd_signal,
            sma_period=args.sma
        )

        if result is not None:
            filename, csv_data = result
        
            print(f"Data with technical indicators successfully saved to {filename}")
            print_llm_prompt(
                args.symbol,
                start_time,
                end_time,
                args.interval,
                filename,
                csv_data
            )
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()