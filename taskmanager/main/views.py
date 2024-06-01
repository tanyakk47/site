import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from django.shortcuts import render
from .forms import TickerForm
from .forms import TickersVol
from .forms import TickersVolume
from .forms import IndicatorForm
def home(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            tickers = form.cleaned_data['tickers'].split()[:10]
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            request.session['tickers_home'] = tickers
            start_date_erl = datetime.strptime(str(start_date), '%Y-%m-%d')
            start_date_erl = start_date_erl - timedelta(days=30)
            start_date_str = start_date_erl.strftime('%Y-%m-%d')
            request.session['start_date_ERL'] = start_date_str
            request.session['start_date_home'] = str(start_date)
            request.session['end_date_home'] = str(end_date)
            imageprice = toImage(Close(plot_indicators(tickers, start_date, end_date)))
            image_base = plot_volatility(tickers, start_date, end_date)
            imagebase = plot_volume(tickers, start_date, end_date)
            request.session['image_base'] = image_base
            request.session['imagebase'] = imagebase
            request.session['imageprice'] = imageprice
            if 'volatility' in request.POST:
                return vol_tickers(request)
            if 'volume' in request.POST:
                return volume_tickers(request)
            if 'indicators' in request.POST:
                return price_tickers(request)
    else:
        form = TickerForm()
    return render(request, 'main/about.html', {'form': form})  # Передано обидві форми у контексті

ind = []
def price_tickers(request):
    global ind
    imagePrice = None
    if 'imageprice' in request.session:
        imagePrice = request.session['imageprice']
    if request.method == 'POST':
        form = IndicatorForm(request.POST)
        if form.is_valid():
            tickers = request.session['tickers_home']
            start_dateerl = request.session['start_date_ERL']
            start_date = request.session['start_date_home']
            end_date = request.session['end_date_home']
            selected_indicator = form.cleaned_data['indicators']
            ind.append(selected_indicator)
            imageprice = toImage(Close(plot_indicators(tickers, start_dateerl, end_date), ind, str(start_date), str(end_date)))
            if 'price' in request.POST:
                return render(request, 'main/indtemp.html', {'imageprice': imageprice, 'form': form})
            if 'reset' in request.POST:
                ind = []
                return render(request, 'main/indicators.html', {'imageprice': imagePrice, 'form': form})
    else:
        form = IndicatorForm()
    return render(request, 'main/indicators.html', {'form': form, 'imageprice': imagePrice})


def toImage(plt):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return image

def Close(df, ind=None, start_date=None, end_date=None):
    close = df['close']
    df['date'] = pd.to_datetime(df['date'])
    date = df['date']
    plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, height_ratios=[10, 5, 5])
    plt.subplot(gs[0])
    if (start_date is not None) and (end_date is not None):
        mask = (date >= start_date) & (date <= end_date)
        date = date[mask]
        close = close[mask]
    plt.grid(True)
    plt.plot(date, close, color='blue', alpha=0.5, label='Close Price')
    if ind is not None and 'sma' in ind:
        MovingAverage(df, mask)
    if ind is not None and 'ema' in ind:
        ExponentialMovingAverage(df, mask)
    if ind is not None and 'bb' in ind:
        BolingerBands(df, mask)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    if ind is not None and 'macd' in ind:
        plt.subplot(gs[1])
        MACD(df, mask)
    if ind is not None and 'rsi' in ind:
        plt.subplot(gs[2])
        RSI(df, mask)
    plt.grid(True)
    return plt

def MovingAverage(df, mask):
    df['date'] = pd.to_datetime(df['date'])
    date = df['date']
    df['MA_7'] = df['close'].rolling(window=7).mean().shift(-6)
    MA = df['MA_7']
    date = date[mask]
    MA = MA[mask]
    return plt.plot(date, MA, color='pink', alpha=1, label='Simple Moving Average')
def ExponentialMovingAverage(df, mask):
    df['date'] = pd.to_datetime(df['date'])
    date = df['date']
    df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean().shift(-6)
    ema = df['EMA_7']
    date = date[mask]
    ema = ema[mask]
    plt.plot(date, ema, color='red', alpha=1, label='Exponential Moving Average')
def BolingerBands(df, mask):
    date = df['date'] = pd.to_datetime(df['date'])
    period = 20
    SMA = df['SMA'] = df['close'].rolling(window=period).mean().shift(-20)
    # Розрахунок стандартного відхилення
    df['STD'] = df['close'].rolling(window=period).std().shift(-20)
    # Розрахунок верхньої та нижньої смуг
    multiplier = 2
    Upper_Band = df['SMA'] + (multiplier * df['STD'])
    Lower_Band = df['SMA'] - (multiplier * df['STD'])
    date = date[mask]
    SMA = SMA[mask]
    Upper_Band = Upper_Band[mask]
    Lower_Band = Lower_Band[mask]
    # Побудова графіку
    plt.plot(date, SMA, label='Середня лінія (SMA)', color='black')
    plt.plot(date, Upper_Band, label='Верхня смуга (Upper Band)', color='black')
    plt.plot(date, Lower_Band, label='Нижня смуга (Lower Band)', color='black')
    return plt.fill_between(date,  Upper_Band, Lower_Band, color='grey', alpha=0.3)


def RSI(df, mask):
    df['date'] = pd.to_datetime(df['date'])
    date = df['date']
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=7).mean().shift(-6)
    loss = -delta.where(delta < 0, 0).rolling(window=7).mean().shift(-6)
    rsi = 100.0 - (100.0 / (1.0 + (gain / loss )))
    date = date[mask]
    rsi = rsi[mask]
    plt.plot(date, rsi, color='blue', alpha=0.5, label='RSI')
    plt.axhline(y=70, color='black', linestyle='--')
    plt.axhline(y=30, color='black', linestyle='--')
    plt.title('\nRSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.tick_params(axis='y', which='both', left='off', right='off')  # Turn off y-axis ticks
    plt.tight_layout()
    plt.legend()
    return plt

def MACD(df, mask):
    exp1 = df['close'].ewm(span=12, adjust=False).mean().shift(-6)
    exp2 = df['close'].ewm(span=26, adjust=False).mean().shift(-6)
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean().shift(-6)
    date = df['date']
    date = date[mask]
    macd= macd[mask]
    signal = signal[mask]
    plt.plot(date, macd, color='black', alpha=1, label='MACD')
    plt.plot(date, signal, color='green', alpha=1, label='Signal')
    plt.title('\nMACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    return plt

def plot_indicators(tickers, start_date, end_date):
    plt.figure(figsize=(13, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(tickers)))
    for ticker, color in zip(tickers, colors):
        query_string = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&datatype=csv&apikey=igGinzXTh9e3txjqjSvM1oZvR3EB6iwh'
    try:
        df = pd.read_csv(query_string)
        if df.empty:
            raise ValueError("The fetched dataframe is empty.")
    except Exception as e:
        return 'Error'
    return df

def volume_tickers(request):
    imagebase = None
    if 'imagebase' in request.session:
        imagebase = request.session['imagebase']
    if request.method == 'POST':
        form2 = TickersVolume(request.POST)
        if form2.is_valid():
            start_date = request.session['start_date_home']
            end_date = request.session['end_date_home']
            tickers_vol = form2.cleaned_data['tickers'].split()[:10]
            image_base_volume = plot_volume(tickers_vol, start_date, end_date)
            if 'volumecompare' in request.POST:
                return render(request, 'main/volumecompare.html', {'form2': form2, 'image_base_volume': image_base_volume})
    else:
        form2 = TickersVolume()
    return render(request, 'main/volume.html', {'form2': form2, 'imagebase': imagebase})
def vol_tickers(request):
    image_base = None
    if 'image_base' in request.session:
        image_base = request.session['image_base']
        # del request.session['image_base']
    if request.method == 'POST':
        form1 = TickersVol(request.POST)
        if form1.is_valid():
            start_date = request.session['start_date_home']
            end_date = request.session['end_date_home']
            tickers_vol = form1.cleaned_data['tickers'].split()[:10]
            image_base_vol = plot_volatility(tickers_vol, start_date, end_date)
            if 'compare' in request.POST:
                return render(request, 'main/volcompare.html', {'form1': form1, 'image_base_vol': image_base_vol})
    else:
        form1 = TickersVol()
    return render(request, 'main/volatility.html', {'form1': form1, 'image_base': image_base})

def plot_volume(tickers, start_date, end_date):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.jet(np.linspace(0, 1, len(tickers)))
    for ticker, color in zip(tickers, colors):
        query_string = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&datatype=csv&apikey=igGinzXTh9e3txjqjSvM1oZvR3EB6iwh'
        df = pd.read_csv(query_string)
        df['date'] = pd.to_datetime(df['date'])
        plt.plot(df['date'], df['volume'], label=f'Volume {ticker}', color=color, alpha=0.75)
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    return toImage(plt)

def rolling_volatility(close_prices, window=21):
    log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    annualized_volatility = rolling_std * np.sqrt(252)
    return annualized_volatility
def plot_volatility(tickers, start_date, end_date):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.jet(np.linspace(0, 1, len(tickers)))
    for ticker, color in zip(tickers, colors):
        query_string = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&datatype=csv&apikey=igGinzXTh9e3txjqjSvM1oZvR3EB6iwh'
        df = pd.read_csv(query_string)
        df['date'] = pd.to_datetime(df['date'])
        volatility = rolling_volatility(df['close'])
        plt.plot(df['date'], volatility, label=f'Volatility {ticker}', color=color, alpha=0.75)
    plt.title('Stock Volatility Comparison')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    return toImage(plt)

