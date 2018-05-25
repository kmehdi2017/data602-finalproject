# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:39:10 2018
Final Project, Data 602
@author:Mehdi Khan
"""


import pandas as pd
import numpy as np
import requests
import datetime

from flask import Flask, render_template, request 

from bokeh.plotting import figure 
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
from bokeh.transform import transform
from bokeh.palettes import  Plasma256


from statsmodels.tsa.arima_model import ARIMA

#Libraries for LSTM
import keras
from keras.models import Sequential 
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle




app = Flask(__name__)
#keras.backend.backend()
# Shows the main page
@app.route("/")
def show_main_page():
    return render_template('main.html', msg = "!! WELCOME !!")

@app.route("/corr")
def show_corelation():
    cocurrencies = ['BTC','ETH','LTC','XRP','VTC','DASH']        
    p = create_corrDF(cocurrencies)
    script, div = components(p)
    plot_url="/static/coinlogo.png"
    return render_template("corr.html", script=script, div=div,  plot_url=plot_url)

@app.route("/submitCorr",methods=['POST'])
def execute_corr():
    selectcurrencies =  request.form.getlist('cryptos')  
    p = create_corrDF(selectcurrencies)
    script, div = components(p)
    plot_url="/static/coinlogo.png"
    return render_template("corr.html", script=script, div=div,plot_url=plot_url)

@app.route("/submitStat_hist",methods=['POST'])
def execute_statHist():
    selectcurrencies =  request.form.getlist('cryptos')  
    p,prestat,poststat = plotdata(selectcurrencies)   
    script, div = components(p)
    precurrency1 = prestat[0]['name']
    precurrency1 = "Pre-Spike Statistics for "+ precurrency1
    predf1 = prestat[0]['df']
    postcurrency1 = poststat[0]['name']
    postcurrency1 = "Post-Spike Statistics for "+ postcurrency1
    postdf1 = poststat[0]['df']
    pre_stat1=predf1.to_html(index=False,classes="statTable")
    post_stat1=postdf1.to_html(index=False,classes="statTable")
    plot_url="/static/coinlogo.png"
    if len(selectcurrencies)>1:
        precurrency2 = prestat[1]['name']
        precurrency2 = "Pre-Spike Statistics for "+ precurrency2
        predf2 = prestat[1]['df']
        postcurrency2 = poststat[1]['name']
        postcurrency2 = "Post-Spike Statistics for "+ postcurrency2
        postdf2 = poststat[1]['df']        
        pre_stat2=predf2.to_html(index=False,classes="statTable")
        post_stat2=postdf2.to_html(index=False,classes="statTable")
    else:
        precurrency2 = ""
        precurrency2 = ""      
        postcurrency2 = ""
        postcurrency2 = ""        
        pre_stat2=""
        post_stat2=""
        
    return render_template("stat_hist.html", script=script, div=div,
                           prestat1=pre_stat1,poststat1=post_stat1,
                           pre_crypto1=precurrency1, post_crypto1=postcurrency1,                           
                           prestat2=pre_stat2,poststat2=post_stat2,
                           pre_crypto2=precurrency2, post_crypto2=postcurrency2,plot_url=plot_url )


@app.route("/stat_hist")
def show_history_stat():
    #default currencies 
    currencies = ['BTC','ETH']    
    p,prestat,poststat = plotdata(currencies)   
    script, div = components(p)
    precurrency1 = prestat[0]['name']
    precurrency1 = "Pre-Spike Statistics for "+ precurrency1
    predf1 = prestat[0]['df']
    postcurrency1 = poststat[0]['name']
    postcurrency1 = "Post-Spike Statistics for "+ postcurrency1
    postdf1 = poststat[0]['df']
    
    precurrency2 = prestat[1]['name']
    precurrency2 = "Pre-Spike Statistics for "+ precurrency2
    predf2 = prestat[1]['df']
    postcurrency2 = poststat[1]['name']
    postcurrency2 = "Post-Spike Statistics for "+ postcurrency2
    postdf2 = poststat[1]['df']
    plot_url="/static/coinlogo.png"
    return render_template("stat_hist.html", script=script, div=div,
                           prestat1=predf1.to_html(index=False,classes="statTable"),poststat1=postdf1.to_html(index=False,classes="statTable"),
                           pre_crypto1=precurrency1, post_crypto1=postcurrency1,
                           
                           prestat2=predf2.to_html(index=False,classes="statTable"),poststat2=postdf2.to_html(index=False,classes="statTable"),
                           pre_crypto2=precurrency2, post_crypto2=postcurrency2, plot_url = plot_url)

@app.route("/prediction")
def show_prediction():
    marketlist = get_Markets()['MarketCurrency']
    plot_url="/static/coinlogo.png"
    return render_template('prediction.html', tickerlist=marketlist,plot_url=plot_url)

@app.route("/submitPredict",methods=['POST'])
def execute_prediction():
    
    marketlist = get_Markets()['MarketCurrency']      
    ticker = request.form['symbol']        
            
    ask,bid,comment = find_price_crypto(ticker)
    
    msg=""
    if ask == 0:
        msg = comment
        return render_template('prediction.html', msg = msg,tickerlist=marketlist)
    else:
        future_price,dockermsg,p = get_predicted_price(ticker)
        askprice= str(ask)
        bidprice= str(bid)            
       
        predictedsymbol = "Predicted closing price for next 7 days"   
        statistics = get_twentyfourhr_stat(ticker)
        #titleStat = "24 hours open, close, low and high price statistics for "+ticker
        mx = statistics.loc[statistics.Statistics=="Maximum price"]["high"]
        mn = statistics.loc[statistics.Statistics=="Minimum price"]["low"]
        n = statistics.loc[statistics.Statistics=="count"]["low"]  
        mx =float(mx)
        mn = float(mn)
        n = float(n)
        av = round(mx+mn/n,2)
        script, div = components(p)
        plot_url="/static/coinlogo.png"
        titleprice = "Selected currency: " +ticker+ " || current ASK Price: $" + askprice + " || current BID Price: $" + bidprice
        titleStat2 = "24 Hour Maximum: "+str(mx)+" || 24 Hour Minimum: "+str(mn)+" || 24 Hour average: "+ str(av)
        #statisticsDF=statistics[1:]
        return render_template('prediction.html',dockerError=dockermsg,predictprice=predictedsymbol,
                               predicted=future_price.to_html(index=False,classes="tradeTbl"),
                               tickerlist=marketlist, price=titleprice,txtStat2=titleStat2,currentSymbol=ticker,
                               script=script, div=div,plottext="The below plots show the predicted price with around three weeks of past price. ",
                               plotins="Please ",plotins1="hover ",plotins2=" over the plot lines to see past or predicted price and associated dates,",
                               plotins3="Zoom tool ", plotins4=" can be used for a closer look.",plot_url=plot_url)
        

def get_marketHistory(name, timespan):
    hist_data = requests.get("https://min-api.cryptocompare.com/data/histoday?fsym="+str(name)+"&tsym=USD&limit="+timespan+"&aggregate=1&e=CCCAGG").json()
    df = pd.DataFrame(hist_data['Data'])  
    df['time'] = [datetime.datetime.fromtimestamp(d) for d in df['time']]  
    postspike = df[df['time']>'2017-04-01']
    prespike = df[df['time']<'2017-04-01']
    return df,prespike,postspike


    
    
 # The project is limited to only the USDT market in the BITTEREX exchange. The get_Markets() function finds the 
# market information and returns only the USDT market information in a dataframe. It uses API provided by BITTEREX
def get_Markets():
    market = requests.get("https://bittrex.com/api/v1.1/public/getmarkets").json()
    marketDF = pd.DataFrame(market['result'])
    marketDF = marketDF.loc[marketDF['BaseCurrency']=='USDT']
    return marketDF
   
# get_twentyfourhr_stat() collects 24 hours trading information of a given currency through API provided by cryptocompare.com
# the crytocompare.com was selected since it collects information from several exchanges and provides a better insight 
# of a currency's trading information and market status. The function returns statistical information of the data
# in a dataframe
def get_twentyfourhr_stat(ticker):    
    hourlydata = requests.get("https://min-api.cryptocompare.com/data/histohour?fsym="+ticker+"&tsym=USDT&limit=24&aggregate=3&e=CCCAGG").json()
    hourlyDF= pd.DataFrame(hourlydata['Data']) 
    hourlyDF = hourlyDF.iloc[:,0:4].describe().reset_index()   
    statDF = round(hourlyDF,2).iloc[[0,1,2,3,7]]
    statDF['index']=  ["count","Mean price","Standard Deviation","Minimum price","Maximum price"]
    statDF = statDF.rename(columns={'index':'Statistics'})    
    return statDF    
    
       
 

def plotdata(currencynames):
         
    colour = ""
    preplot_list = []
    postplot_list = []
    preStat_list =[]
    postStat_list =[]
    
    for currency in currencynames:
         if (currency == 'BTC'):
              colour = 'navy'
         elif (currency == 'ETH'):
             colour = 'brown'
         elif (currency == 'LTC'):
             colour = 'blueviolet'
         elif (currency == 'XRP'):
             colour = 'coral'
         elif (currency == 'VTC'):
             colour = 'cyan'
         elif (currency == 'DASH'):
             colour = 'darkcyan'
         
         dffull,dfpre,dfpost = get_marketHistory(currency, '730')
         pre = plot_currency(dfpre,currency,colour,"Pre-spike")
         post = plot_currency(dfpost,currency,colour,"Post-spike")
         stat_pre = get_market_stat(dfpre)
         stat_post = get_market_stat(dfpost)
         preplot_list.append(pre)
         postplot_list.append(post)
         
         preStat_list.append({'name':currency,'df':stat_pre})
         postStat_list.append({'name':currency,'df':stat_post})
         
    grid = gridplot([preplot_list,postplot_list])
         
    return grid,preStat_list,postStat_list

def plot_currency(df,symbol,col,timeframe):   
        
    
    p = figure(width=600, height=350, x_axis_type="datetime", tools=['box_zoom','reset'] )
   
    df['h-l'] = df.high- df.low 
    # getting 30 days moving window for close price
    window_size = 30
    window = np.ones(window_size)/float(window_size)
    moving_avg = np.convolve(df.close, window, 'same')   
    
    src = ColumnDataSource(data = dict(time=df['time'],close=df['close'], open=df['open'],hl=df['h-l'],ma=moving_avg))
    src.add(df['time'].apply(lambda d: d.strftime('%m/%d/%Y')), 'date')
    h1 = p.circle(x='time',y='close', size=4, color='darkred', alpha=0.2, legend=symbol+' close price', source=src)
    hover1 = HoverTool(renderers=[h1],tooltips=[ 
    ( 'date',   '@date'       ),   
    ("close", "@close{$ 0,0.00}"),    
    ] ,   )
    p.add_tools(hover1)

    h2 = p.line(x='time',y='ma', color='darkgreen', line_width=2, legend= symbol+' mov-avg', source=src)
    hover2 = HoverTool(renderers=[h2],tooltips=[ 
    ( 'date',   '@date'       ),   
   ("mov-avg:", "@ma{$ 0,0.00}"),
    ] ,   )
    
    p.add_tools(hover2)
    
    h3 = p.line(x='time',y='hl', color='purple', line_width=1, legend= symbol+' high-low', source=src, name='high-low')
    hover3 = HoverTool(renderers=[h3],tooltips=[ 
    ( 'date',   '@date'       ),   
     ("hi-lo:", "@hl{$ 0,0.00}"),    
    ], )
    p.add_tools(hover3)
        #closeDF[]
    p.title.text = timeframe+" 30 days moving average, close and high-low price"
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.grid.grid_line_alpha = 0.1
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.ygrid.band_fill_color = col
    p.ygrid.band_fill_alpha = 0.1    
    p.legend.click_policy="hide" 
     
    
    return p 


def get_market_stat(df):       
    stdf = df.iloc[:,0:4].describe().reset_index()   
    #stdf = df.iloc[:,[0,1,2,3,6]].describe().reset_index()   
    #statDF = round(hourlyDF2,2).iloc[[0,1,2,3,7]]
    statDF = round(stdf,2)
    statDF['index']=  ["count","Mean price","Standard Deviation","Minimum price",'Q1','Q2','Q3',"Maximum price"]
    statDF = statDF.rename(columns={'index':'Statistics'})    
    return statDF

# The find_price_crypto function takes one parameters: the ticker . The function uses Bitterex API to 
# retrieve cryptocurrency price  from the BITTEREX exchange. This function collects data only from currencies in USDT market. If for any reason the price information is not available 
# it show a message asking to try trading later. After retrieving the price information it converts the price into float. It
# finally returns the price value ( price = 0 if no valid price is found) along with a message string .  
def find_price_crypto(name): 
    
    url="https://bittrex.com/api/v1.1/public/getticker?market=USDT-"+ name
   
    msg=""
    
    if (name is None) or (name == "") :
        price = 0
        msg = "Symbol or a trading option was not entered....please check !!"
    else:
        req = requests.get(url).json()
        if req['success'] == True:   
            askprice = req['result']['Ask']          
            bidprice = req['result']['Bid']
            try:
                if type(askprice) != float:
                    price = float(price) 
                if type(bidprice) != float:
                    price = float(price)                     
            except ValueError:
                msg = "Price is NOT available!!!"   
            
        else:
            msg = "Price is NOT available!!!"
            askprice = 0
            bidprice = 0
    
    priceinfo = round(askprice,2),round(bidprice,2), msg     
       
    return priceinfo

def create_corrDF(currencies):
     #currencies = ['BTC','ETH','LTC','XRP','VTC','DASH']
     
     datafull,dataPre,dataPost = get_marketHistory(currencies[0], '730')    
     corrDFpre = dataPre[['close']]
     corrDFpre = corrDFpre.rename(int,{'close':currencies[0]})
     
     corrDFpost = dataPost[['close']]
     corrDFpost = corrDFpost.rename(int,{'close':currencies[0]})
     
     #corrDFpre = corrDFpre.loc[:,'close']
     if len(currencies)>1:
         for currency in currencies:
             dffull,dfpre,dfpost = get_marketHistory(currency, '730')
             corrDFpre.loc[:,currency] = dfpre.loc[:,'close']
             corrDFpost.loc[:,currency] = dfpost.loc[:,'close']
    
     mapper = LinearColorMapper( palette=Plasma256, low=0, high=1) 
     
     color_bar = ColorBar( color_mapper=mapper,
        location=(-20, 0),
        ticker=BasicTicker(desired_num_ticks=10))
     
     PreCorr = create_corrplot(corrDFpre,'Pre-Spike',mapper,450)
     PostCorr = create_corrplot(corrDFpost,'Post-Spike',mapper,400)
     PreCorr.add_layout(color_bar, 'left')
     grid = gridplot([[PreCorr,PostCorr]])
     
     #plotrow = row(PreCorr,PostCorr)
     
     return grid
   
   
def create_corrplot(df,timeframe,mpr,wd):
    df = df.pct_change()
    df = df.corr()

    df.index.name = 'allcurrenciesX'
    df.columns.name = 'allcurrenciesY'
    
    #formatting the dataframe
    df = df.stack().rename("value").reset_index()    
    
    #mapper = LinearColorMapper( palette=PuBu[9], low=0, high=1) 
    
    
    Hover = HoverTool(tooltips=[ 
    ( 'corr-coefficient',   '@value' ),
    ] ,  )
    
    toolstoadd = "box_zoom","reset"
    p = figure(
        tools=toolstoadd,
        plot_width=wd,
        plot_height=400,
        title="Correlation plot for "+timeframe+" period",
        x_range=list(df.allcurrenciesX.drop_duplicates()),
        y_range=list(df.allcurrenciesY.drop_duplicates()),
        toolbar_location="above",
        x_axis_location="below")
    p.add_tools(Hover)
    
    p.rect(
        x="allcurrenciesX",
        y="allcurrenciesY",
        width=1,
        height=1,
        source=ColumnDataSource(df),
        line_color=None,
        fill_color=transform('value', mpr))
    return p
    

################ Machine learning algorithms######################

# TimeSeries Analaysis with ARIMA and LSTM RNN models

# collection of two years historic data  

# make a timeseris enabled dataset 
def get_ts(df): 
    historyDF = df
   # historyDF.time = historyDF.time.apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%m/%d/%Y'))
   # historyDF.time=pd.to_datetime(historyDF['time'], format='%m/%d/%Y')
    historyDF = historyDF.set_index('time')
    historyDF = historyDF[~(historyDF['close'] == 0)]
    return historyDF.close



### Decision making process with Time Series data
# The following functions were created to analize and visualize the data, the trend, and check stationarity, ways of finding and eliminating trends
# and finally find the right model (in this case ARIMA)

      
# After checking several process it was decided that data will be log transformed so the difference between higher and lower values are atjusted, then differencing 
# will be used to smooth the data, a 15 days (two weeks) lag was used. The  transformation_differencing functiobn take timeseried data
# and do the job as mentioned above abd return a log transformed data with difference in time lags
      
def transformation_differencing(ts):
    data_log =np.log(ts)
    difference = data_log - data_log.shift(1)
    difference.dropna(inplace=True)
    return difference, data_log      


# The ARIMA model and forecasting
# The forecastpriceARIMA function creates an ARIMA model and forecast the closing price of the 
# currency for the next 7 days
def forecastpriceARIMA(tsdata,p,q):
    model = ARIMA(tsdata, order=(p, 1,q))  
    modelResult= model.fit(disp=-1)  
    next_dates = [ tsdata.index[-1] + datetime.timedelta(days=i) for i in range(7) ]
       
    forecast = pd.Series(modelResult.forecast(steps=7)[0],next_dates)
    forecast = np.exp(forecast)
    return forecast
 
    

############# LSTM RNN
## normalization of data with Min-Max scaling. With this scaling the crypto currency closeing price data 
# is scaled  to a fix range of 0 to 1. This is an alternative of z-score standardization although
# it can potentially cause suppresion of outlier effects because of smaller standard deviation.
# typical neural network algorithm require data that on a 0-1 scale
scaler = MinMaxScaler(feature_range=(0, 1))

# the moving_window takes a number (window size) and a dataframe as its parameters. It shifts 
# column by 1 each time until it reaches the number of times specfied in the window size
# and then concatenate the shifted column to the original data  and finally returns  a datafrme with number of columns 
#as specified by the window size plus one. 

# The concept:  Throgh this process a long series can be sliced intoto smaller sizes. The benefit
# of doing this isto reduce the length of the sequence. And it can be very useful by customizing the 
# number of previous timesteps to predict the current timestep that can give a LSTM model a better 
# learning experience.
# In this context a 30 days windows were considered. i.e 30 days of price data would be used in every single 
# input (X) to predict the 31nd day price (y) and so on. 
def moving_window(windowsize,ds):
    copy_ds = ds.copy()
    for i in range(windowsize):
        ds = pd.concat([ds, copy_ds.shift(-(i+1))], axis = 1)
    ds.dropna(axis=0,inplace=True)
    return ds

# The scaleddata function scaled down the input data beetween 0 to 1 
def scaleddata(ds):    
    dataset = scaler.fit_transform(ds.values)
    dataset = pd.DataFrame(dataset)
    return dataset

#  create_model_dataset returns 4 sets of dataset (train X,y and Test X,y), with 80% are for training the model and
# 20% for testing the model. 
def create_model_dataset(df,ratio =.8):
    size = round(df.shape[0] * ratio)
    train = df.iloc[:size,:]
    test = df.iloc[size:,:]
    train = shuffle(train)
    trainx = train.iloc[:,:-1].values
    trainy = train.iloc[:,-1].values
    testx = test.iloc[:,:-1].values
    testy = test.iloc[:,-1].values
    return trainx,trainy,testx,testy

# create_lstm_model creates the LSTM RNN model. 
# A double stacked LSTM layers are used, by setting return_sequences = True it was ensured that
# For every input to the first layer an output will be fed to second LSTM layer. 
def create_lstm_model(data, activation="linear",l="mse",opt="adam"):
    model = Sequential()
    model.add(LSTM(input_shape = (data.shape[1],data.shape[2]), output_dim= data.shape[1], return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation(activation))
    model.compile(loss=l, optimizer=opt)
    #model.summary()
    return model
    
    
# sevendays_forecast function predict futre 7 days' prices. Basically the model predicts one at 
# a time for seven times, after each prediction the predicted data is added back to to the input dataset 
# (and the oldest data is removed from the stack) to make prediction for the next timestep.
def sevendays_forecast(testx, model):
    forecasts =[]
    pred_data = [testx[0,:].tolist()]
    pred_data = np.array(pred_data)
    for i in range(7):
        prediction = model.predict(pred_data)
        forecasts.append(prediction[0,0])
        prediction = prediction.reshape(1,1,1)
        pred_data = np.concatenate((pred_data[:,1:,:],prediction),axis=1)
    
    forecasts = np.array(forecasts)
    forecasts = forecasts.reshape(-1,1)
    forecasts = scaler.inverse_transform(forecasts)    
    return forecasts

# all_future_price function add all the predictions done by both models
# and returns a dataframe of the combined data. 
def all_future_price(arima,lmts,currency):
     ds = pd.DataFrame(arima)
     ds.index = ds.index.strftime('%m/%d/%Y')
     ds.columns=[currency+' ARIMA Price']
     ds.iloc[:,0] = round(ds.iloc[:,0],3)
     if lmts != 'NA':    
        lmts = lmts.astype('float64')
        ds[currency+' LSTM Price'] = lmts
        ds.iloc[:,1] = round(ds.iloc[:,1],3) 
        msg = ""
     else:
        msg = "LSTM model prediction could not be available because of DOCKER issue"
         
     ds = ds.reset_index()
     ds = ds.rename(columns={'index':'Prediction Dates'})        
     return ds, msg 
    

# get_predicted_price basically calls the previous function to create the historic
# dataset, create the LSTM model and finally predict the data along with all the 
# required intermediate steps

def get_predicted_price(name):
# collection of two years historic data 
    
    full,pre,df = get_marketHistory(name, '730')

#########ARIMA model steps###################
    TS = get_ts(df)
    TSDiff, data_log = transformation_differencing(TS) 
    ARIMA_forecast = forecastpriceARIMA(data_log,0,1)



#LSTM RNN model steps
    LSTMds = TS.reset_index()
# no need for datetime data
    LSTMds = LSTMds.drop('time', axis=1)

# scale data
    LSTM_dataset = scaler.fit_transform(LSTMds.values)
    LSTM_dataset_scaled = pd.DataFrame(LSTM_dataset)

    LSTM_window = moving_window(30,LSTM_dataset_scaled)
    TrainX,TrainY,TestX,TestY = create_model_dataset(LSTM_window)

# reshape input to be [samples, time steps, features]
    TrainX = np.reshape(TrainX, (TrainX.shape[0], TrainX.shape[1], 1))
    TestX = np.reshape(TestX, (TestX.shape[0], TestX.shape[1],1))
    
    

    try:        
        LSTM_model = create_lstm_model(TestX, activation="linear",l="mse",opt="adam")
    except ValueError:         
        forecasted_price = all_future_price(ARIMA_forecast,'NA',name)            
        return forecasted_price  
    else:
        LSTM_model.fit(TrainX,TrainY,batch_size=512,epochs=3,validation_split=0.1)
        predicts = LSTM_model.predict(TestX)
        predicts = scaler.inverse_transform(predicts)
    
        LSTM_future_price = sevendays_forecast(TestX,LSTM_model)
    
        forecasted_price,msgstr = all_future_price(ARIMA_forecast,LSTM_future_price,name)
        keras.backend.clear_session()
        p = plot_prediction(df,forecasted_price)
        return forecasted_price, msgstr, p
    



def plot_prediction(df,predicted):   
    
    p = figure(width=600, height=350, x_axis_type="datetime", tools=['box_zoom','reset'] )
    p1 = figure(width=600, height=350, x_axis_type="datetime", tools=['box_zoom','reset'] )
    #p = figure(width=450, height=300, x_axis_type="datetime")
    DF1 = df[-30:]
    date =pd.DatetimeIndex(DF1.time) + pd.DateOffset(6)
    date = date.to_series()
    cl = df[-23:]['close']
   
    f_dates = predicted[predicted.columns[0]]
    f_dates = pd.to_datetime(f_dates)
    AR =  predicted[predicted.columns[1]]
    ls = predicted[predicted.columns[2]]
    src = ColumnDataSource(data = dict(time=date,close=cl, future_dates=f_dates,ARIMA = AR,LSTM=ls))
    src.add(df['time'].apply(lambda d: d.strftime('%m/%d/%Y')), 'date_')
    src.add(f_dates.apply(lambda d: d.strftime('%m/%d/%Y')), 'date2')
#    
    h1 = p.line(x='time',y='close', color='darkgreen', line_width=1, legend= 'past price', source=src)
    hover1 = HoverTool(renderers=[h1],tooltips=[ 
    ( 'date',   '@date_'       ),   
   ("price:", "@close{$ 0,0.00}"),
    ] ,   )
    
    p.add_tools(hover1)
    
    h2 = p.line(x='future_dates',y='ARIMA', color='purple', line_width=2, legend= 'ARIMA price', source=src, name='arima')
    hover2 = HoverTool(renderers=[h2],tooltips=[ 
    ( 'date',   '@date2'       ),   
     ("ARIMA:", "@ARIMA{$ 0,0.00}"),    
    ], )
    p.add_tools(hover2)
        #closeDF[]
    p.title.text = " Predicted price by ARIMA model"
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.grid.grid_line_alpha = 0.1
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.ygrid.band_fill_color = 'navy'
    p.ygrid.band_fill_alpha = 0.1    
    p.legend.click_policy="hide" 
    
    
    h3 = p1.line(x='time',y='close', color='darkgreen', line_width=1, legend= 'past price', source=src)
    hover3 = HoverTool(renderers=[h3],tooltips=[ 
    ( 'date',   '@date_'       ),   
   ("price:", "@close{$ 0,0.00}"),
    ] ,   )
    
    p1.add_tools(hover3)
    
    h4 = p1.line(x='future_dates',y='LSTM', color='purple', line_width=2, legend= 'LSTM price', source=src, name='lstm')
    hover4 = HoverTool(renderers=[h4],tooltips=[ 
    ( 'date',   '@date2'       ),   
     ("LSTM:", "@LSTM{$ 0,0.00}"),    
    ], )
    p1.add_tools(hover4)
        #closeDF[]
    p1.title.text = " Predicted price by LSTM model"
    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"
    p1.grid.grid_line_alpha = 0.1
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'
    p1.ygrid.band_fill_color = 'coral'
    p1.ygrid.band_fill_alpha = 0.1    
    p1.legend.click_policy="hide"
     
    grid = gridplot([[p,p1]])
    
    return grid

     

          
if __name__ == '__main__':
     app.run(host='0.0.0.0')
     