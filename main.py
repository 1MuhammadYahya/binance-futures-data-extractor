#!/usr/bin/env python3

import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
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
            start_dt = datetime.utcfromtimestamp(start_time / 1000)
            end_dt = datetime.utcfromtimestamp(end_time / 1000)
            start_str = start_dt.strftime("%d:%H:%M")
            end_str = end_dt.strftime("%d:%H:%M")
            filename = f"{symbol}_{interval}_{start_str}-{end_str}.csv".replace(':', '')

        # Write to CSV, including only the columns we need
        columns_to_save = [
            'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'sma_20'
        ]
        
        df_final[columns_to_save].to_csv(filename, index=False)

        print(f"Successfully saved {len(df_final)} records to {filename}")
        print(f"Technical indicators included: RSI-14, MACD-12/26/9, SMA-20")
        return filename

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

def main():
    parser = argparse.ArgumentParser(description='Fetch historical data from Binance Futures with technical indicators.')
    parser.add_argument('-symbol', required=True, help='Trading pair symbol (e.g., SOLUSDT)')
    parser.add_argument('-tframe', required=True, type=int, help='Time frame in minutes (how far back from end time)')
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
        start_time = end_time - (args.tframe * 60 * 1000)
        
        filename = fetch_historical_data_with_indicators(
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
        
        if filename:
            print(f"Data with technical indicators successfully saved to {filename}")
            print(f"Prompt: \nHere is the data for the {args.symbol} Binance Futures pair. The data starts from time {start_time} and ends at {end_time}. Each data point is over a {args.interval} period. Help analyze the data. Point out general trends, recent supports and resistances as well as entry and exit strategies to get a decent profit. I am up for both long and short positions. Point out conditions that I should look for before diving into a trade and most importantly, if you feel like it is not the correct time or the trend is too noisy such that it would be too risky to trade, do point that out. I am looking for around 30%-50% profit so let me know how much leverage I should use to be safe enough and reach my goals. Thanks")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()