import yfinance as yf
import pandas as pd
import re

def load_data(ticker="AAPL"):

    asset=ticker.upper()
    asset = asset.replace(" ", "")

    #ticker regex
    pattern = r'^[A-Za-z0-9\.\-]+$'
    if not re.match(pattern, ticker):
        print("Nieprawidłowy ticker. Ticker powinien zawierać tylko litery")
        exit()

    data = yf.Ticker(asset).history(period='1d', interval='1m')

    data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['Anomaly'] = (data['Close'] > data['EMA30']) & (data['Close'].shift(1) <= data['EMA30'].shift(1))
    
    anomalies = data[data['Anomaly']]

    return data,anomalies

    #asset_data['EMA30'] = asset_data['Close'].ewm(span=30, adjust=False).mean()

if __name__ == "__main__":

    answer = input("Podaj ticker: ")
    if answer=='':
        print("Nie podano nazwy")
        exit()  

    stock_data = load_data(answer)
    print(stock_data)