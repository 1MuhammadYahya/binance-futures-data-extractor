# Binance Futures Historical Data & Indicator Fetcher

A **free and open-source Python tool** for fetching historical kline (candlestick) data from **Binance Futures**, calculating popular **technical indicators** (RSI, MACD, SMA), and exporting the results to a CSV file. The tool also generates **LLM-ready prompts** for market analysis and can copy them directly to your clipboard.

---

## **Features**
- Fetch historical data for any **Binance USDT-M Futures trading pair**.
- Calculate **RSI, MACD, and SMA** indicators with fully configurable parameters.
- Automatically export data to **CSV** with readable timestamps.
- Generate **AI-ready trading prompts** with clipboard integration.
- **Nix-based development shell** for fast, reproducible setup.

---

## **Requirements**
- Python **3.8+**
- Binance API Python SDK
- NumPy, Pandas, Pyperclip

Install dependencies (if not using Nix):
```bash
pip install -r requirements.txt
```

---

## **Quick Start with Nix**

If you use **Nix** (recommended for a reproducible environment), you can quickly set up the project:

```bash
nix-shell
```

This will:

* Create and activate a Python virtual environment (`venv`).
* Install all required dependencies including:

  * `numpy`
  * `pandas`
  * `pyperclip`
  * `binance-futures-connector==4.1.0`
  * `requests`
* Provide tools like `gcc` and `xclip` for clipboard support.
---

## **Usage**

### **Basic Command**

```bash
./fetch_data.py -symbol SOLUSDT -tframe 1000
```

Fetches the last **1000 1-minute candles** for `SOLUSDT` and calculates RSI, MACD, and SMA.

---

### **Available Arguments**

| Argument       | Description                                                               | Default  |
| -------------- | ------------------------------------------------------------------------- | -------- |
| `-symbol`      | Trading pair symbol (e.g., `BTCUSDT`, `ETHUSDT`). **Required**.           | None     |
| `-tframe`      | Number of intervals (candles) to fetch. **Required**.                     | None     |
| `-interval`    | Interval between data points (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, etc.). | `1m`     |
| `-endtime`     | End time (epoch timestamp or datetime string like `2025-07-20 15:30`).    | Now      |
| `-filename`    | Custom CSV filename.                                                      | Auto-gen |
| `-rsi`         | RSI period.                                                               | `14`     |
| `-macd_fast`   | Fast period for MACD.                                                     | `12`     |
| `-macd_slow`   | Slow period for MACD.                                                     | `26`     |
| `-macd_signal` | Signal period for MACD.                                                   | `9`      |
| `-sma`         | SMA period.                                                               | `20`     |

---

### **Examples**

**Fetch last 500 15-minute candles for BTCUSDT:**

```bash
python ./fetch_data.py -symbol BTCUSDT -tframe 500 -interval 15m
```

**Fetch 1-hour candles for ETHUSDT ending at a specific time:**

```bash
python ./fetch_data.py -symbol ETHUSDT -tframe 200 -interval 1h -endtime "2025-07-20 15:30"
```

**Custom filename and indicators:**

```bash
python ./fetch_data.py -symbol SOLUSDT -tframe 300 -interval 5m -filename sol_analysis.csv -rsi 10 -macd_fast 8 -macd_slow 21 -sma 50
```

---

## **LLM Prompt Generator**

After fetching data, the script can:

* Generate **custom trading prompts** for AI analysis.
* Copy data, prompts, or both directly to the clipboard.
* Delete the generated CSV file if desired.

---

## **License**

Licensed under the **MIT License**.
You are free to use, modify, and distribute it with proper attribution.

---

## **Contributing**

* Fork the repo
* Create a feature branch
* Submit a PR with clear descriptions

---

## **Future Plans**

* Add **Bollinger Bands**, EMA, and VWAP indicators.
* Add **real-time data streaming**.
* Docker image for containerized deployment.

---

## **Author**

Maintained by **\[Muhammad Yahya A.K.A Yahoo Warraich/ 1MuhammadYahya]**.
PRs and issue reports are welcome!
