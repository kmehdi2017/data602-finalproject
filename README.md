# data602-Final Project
# NOTE:
The project depends on cryptocomapre.com. The tools might fail if the cryptocompare API fails to provide data. The cryptocompare API failed during the test and because of the time limitation this problem was not addressed in the codes. 

Accessing Docker Image from Docker Hub:
---------------------------------------
    docker pull kmehdi2017/data602:finalproject

Group:
---------------------------------------
Mehdi Khan, Ann Liu-Ferrara, David Quarshie

The Project
----------------------------
This study takes a look at cryptocurrency pricing - presented on a python/flask-based application where information can be accessed through various visualizations, data and charts. Cryptocurrencies are exchanged online using Blockchain technology, allowing peer to peer trading with users receiving their funds as fast as 2 hours. GDAX (gdax.com), coinmarket.com, cryptocompare.com, and other various exchanges currently allow users to buy and sell these currencies. In addition to their own platforms all these websites grant access to their APIs with varying capacities, which can be used for insights into different trades at different times. 

This project used APIs from cryptocompare.com and bittrex.com along with the data analysis, statistical and ML tools of Python. The web application and visualizations were done with Bokeh, jQuery, and the Flask library in Python.


The scope, limitations and assumptions:
----------------------------------------
The project study was limited to six cryptocurrencies: BTC, ETH, LTC, XRP, VTC, and DASH (although the flexibility of the procedures and developed tools allow any cryptocurrency to be studied if needed). These six cryptos were selected based on their popularity, market volume and availability of historical data for longer periods. The historical data was collected from cryptocompare.com through their API. Crytocompare.com was selected since it collects information from several exchanges and was assumed to provide a better insight of a currency's trading information and market status. Bittrex API was used to get the currency list.

The cryptocurrency market experienced an abrupt spike in around mid-2017, which attracted many but also concerned others about their existence, potentials and risks. Since these two periods are very distinct, all the studies were done in the context of pre-spike and post-spike time periods of cryptocurrency. The project studied two years of historical data and time after April 1, 2017 was considered post-spike period.

Initial screen:
---------------
At the start of the program the user is greeted with a welcome screen with the project information. it displays buttons with link to specific functionalities of the application. These buttons are common in all the pages that help users to navigate to any specfic page or functionality from any other page. Clicking the “Correlation” button takes a user to the correlation analysis page(corr.html), similarly “History & Statistics” and “Prediction” takes a user to their respective pages to see and compare historical data, statistics and predicted currency price.The "Main" takes users back to the main initial page. Followig image shows part of the initial view:

![image](https://user-images.githubusercontent.com/25092754/40553373-70c8ba3e-6010-11e8-8e0d-5ca96cfbf362.png)

Correlation
------------
In this page users can see default or generate their own corrleation analysis and plots. By having correlation plots side by side users can compare and see if the currencies reacted to market forces in the same way in pre and post-spike periods. It was found in the study that the currencies were more strongly correlated in the post-spike period. Following is a partial view of the correlation page:

![image](https://user-images.githubusercontent.com/25092754/40553789-b789a70c-6011-11e8-9cad-d8d9e0a9ab24.png)

History & Statistics
---------------------
History & Statistics page show the market statistics of crypto currencies and interactive plots of close price, market flactuation (high - low) and 100 day moving average both for the pre and post-spike periods. While the default currencies are shown on page load, users can generate statistics and plots of currencies of their choice. See screenshotsbelow:

![image](https://user-images.githubusercontent.com/25092754/40554120-b620d6c8-6012-11e8-98ba-51b35af9cca0.png)

![image](https://user-images.githubusercontent.com/25092754/40554161-d5470586-6012-11e8-8a85-18a8bb52b529.png)


Prediction
---------------------
Prediction page shows the predicted price of a user selected currency from two models (ARIMA and LSTM) both in table format and in interactive plots along with their current market price and market information. Below is a partial view of the prediction page:

![image](https://user-images.githubusercontent.com/25092754/40554430-ae4baf12-6013-11e8-962d-d37430374b48.png)

The Functions:
---------------
Following are a description of the functions used in the system:

    1. @app.route("/")
    def show_main_page():
Shows the main page

    2. @app.route("/corr")
    def show_corelation():
Shows the correlation page, it calls create_corrDF() function

     3. @app.route("/submitCorr",methods=['POST'])
      def execute_corr():
  Get the user selections from the form submission and then call create_corrDF() function to generate new correlation analysis
  and plots. 
  
      4. @app.route("/submitStat_hist",methods=['POST'])
          def execute_statHist():
  Get the user selection from form submission and calls function plotdata() to generate statistics and plots based on users' selections.

      5. @app.route("/stat_hist")
      def show_history_stat():
  Show the History and Statistics page. Calls plotdata() to generate statistics and plots for default currencies. 
  
      6. @app.route("/prediction")
      def show_prediction():
  Shows the prediction page. Calls get_Markets() function to generate a list of currencies that the users can select from.
  
      7. @app.route("/submitPredict",methods=['POST'])
    def execute_prediction():
Get the user selection from form submission and generate predicted price, plots and current market information of the user selected currency. Calls get_Markets(), find_price_crypto(), get_predicted_price() and get_twentyfourhr_stat()
    
    8. get_marketHistory(name,timespan):
The function uses cryptocompare.com  to get the historical information of a given currency through API provided by cryptocompare.com the crytocompare.com was selected since it collects information from several exchanges and provides better insights of a currency's trading information and market status. The function returns historical data based on the time span value supplied as a parameter.

    9. get_Markets():
  The function finds the market information and returns only the USDT market in a dataframe. It uses API provided by BITTEREX
  
    10. get_twentyfourhr_stat(ticker)
  The function collects 24 hours trading information of a given currency through API provided by cryptocompare.com. The function returns   statistical information of the data in a dataframe.
  
    11. plotdata(currencynames)
  The function takes one or two currencies as its parameter and return a gridplot object (Bokeh library) containing the plots of the supplied currencies in pre and post spike periods. It plots close price, 100 day moving average and high-low price of the currencies.
  
    12. get_market_stat(df)
  The function takes a dataframe of price history and returns a dataframe with price statistics.
  
     13. find_price_crypto(name)
The find_price_crypto function takes one parameters: the ticker . The function uses Bitterex API to retrieve cryptocurrency price  from the BITTEREX exchange. This function collects data only from currencies in USDT market. If for any reason the price information is not available it show a message asking to try trading later. After retrieving the price information it converts the price into float. It
finally returns the price value ( price = 0 if no valid price is found) along with a message string .  

    14. create_corrplot(df,timeframe,mpr,wd)
The function takes a dataframe, timeframe information (pre or post) a mapper object (bokeh) and a number as parameters, creates correlation and then formats the dataframe to create the plots. It returns a figure object (bokeh).

    15. get_ts(df)
 This function generates time-series enabled date from the historical data
 
    16. transformation_differencing(ts)
 This function takes the time series data and apply log transformation and differencing techinques to be used in ARIMA model
 
    17 forecastpriceARIMA(tsdata,p,q)
 The forecastpriceARIMA function creates an ARIMA model and forecast the closing price of the currency for the next 7 days
 
    18. moving_window(windowsize,ds)
the moving_window takes a number (window size) and a dataframe as its parameters. It shifts column by 1 each time until it reaches the number of times specfied in the window size and then concatenate the shifted column to the original data and finally returns a datafrme with number of columns as specified by the window size plus one.

The concept: Throgh this process a long series can be sliced intoto smaller sizes. The benefit of doing this isto reduce the length of the sequence. And it can be very useful by customizing the number of previous timesteps to predict the current timestep that can give a LSTM model a better learning experience.

    19. scaleddata(ds)
The scaleddata function scaled down the input data beetween 0 to 1, a requirement for RNN models

    20. create_model_dataset(df,ratio =.8)
Create_model_dataset returns 4 sets of dataset (train X,y and Test X,y), with 80% of the data for training the model and 20% for testing the model

    21. create_lstm_model(data, activation="linear",l="mse",opt="adam")
create_lstm_model creates the LSTM RNN model. A double stacked LSTM layers are used, by setting return_sequences = True it was ensured that for every input to the first layer an output will be fed to second LSTM layer.

    22. sevendays_forecast(testx, model)
sevendays_forecast function predict futre 7 days' prices for LSTM. Basically the model predicts one at a time for seven times, after each prediction the predicted data is added back to to the input dataset (and the oldest data is removed from the stack) to make prediction for the next timestep.

    23. all_future_price(arima,lmts,currency)
all_future_price function add all the predictions done by both models and returns a dataframe of the combined data.
 
    24. get_predicted_price(name)
get_predicted_price basically calls the previous functions to create the historic dataset, create the LSTM model and finally predict the data along with all the required intermediate steps
 
 
    25. plot_prediction(df,predicted)
Takes price history dataframe and the predicted price (dataframe) as parameters and then plots the predicted price along with around three weeks of past price.  

  




