import yfinance as yf
import pandas as pd


cr_assets = []
cr_assets_ta = []
errors = []

#input usera narazie tak

# answer = input("Podaj coin (np. BTC): ")
# if answer=='':
#     print("Nie podano nazwy")
#     exit()  
# answer=answer.upper()
# cr_assets.append(answer+'-USD')
# cr_assets_ta.append('BINANCE:'+answer+'USD')

# if len(cr_assets) == 0: #brak inputu jakiegokolwiek
#     print("Nie podano nazwy")
#     exit()

#asset = cr_assets[0]
stock_data = yf.Ticker("AAPL").history(period='1y', interval='1d')
print(stock_data)

stock_data['EMA30'] = stock_data['Close'].ewm(span=30, adjust=False).mean()