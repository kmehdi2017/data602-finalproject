# data602-finalproject

Accessing Docker Image from Docker Hub:
---------------------------------------

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
History & Statistics page show the market statistics of crypto currencies and plots of close price, market flactuation (high - low) and 100 day moving average both for the pre and post-spike periods. While the default currencies are shown on page load, users can generate statistics and plots of currencies of their choice. See a partial view below:





