# Project Description
Stock prince forecasting is one of the most critical issues during the stock market trading, the traditional approaches make the time-series  prediciton still challenging.
With development of artificial intelligence, high performance algorithms make reliable prediction possible from the data perspective.\
In this project, we developed  Long Short-Term Memory (LSTM) neural network for time-series forecasting Ford Stock prices in Python with the Keras deeplearning network.\
This problem where given a year and month, the task to predict the Ford Stock price.\
The data source is imported from [Yahoo Finance](https://query1.finance.yahoo.com/v7/finance/download/Fperiod1), the data ranges from 22/07/2019 to 20/07/2020.

## Libraryies & Packages
 - jupyter notebook google colab
 - Keras RNN API,  LSTM
 - tesnsorflow v 10.0

## Installation
```
import pandas as pd
import numpy as numpy
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.processing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import sequential 
from tensorflow.keras.layes import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping 
```

## Data Preprocessing
 We load the data as dataframe, it consist of 252 entries, total of six columns, mermory storage is 13.6 KB, and the test size is 14.\
 LSTMs are sensitive to the scale of the data input. it can be a good practice to rescale the data to the range of 0-10, also called normalizing.\
 We easily normalise the dataset using the MinMaxScaler preprocessing class from the scikit-learn library.\
 
 ## Training & Testing
 With the time-series data, the seqquence values is important. A simple methof that we can use and lenght of test and train datatset and to calcuale the\
 the indect of split point and seperate the data into trianing datasets into 234 periods of the total periods that we can use to train our model, leaving the remaining\
 18 periods for testing the model.\
 
 ## LSTM Model
 The stock prices changes with time and other influential factors.
 It is suitable to utilise LTM nodel to explore the market mechanism from the relevant data.\
 The architecture of the proposed LSTM model is deployed. the dimension of input data is decided according to the number of influential factors of stock price.\
 The LSTM layer is embedded to meomrise and extract containing information from input data, where the ADAM optimizer is used to update the weights and bias and the mean\
 square error is set as the fitness function.\
 The final output is given by a fully connected layer after the termination condition is reached.
 
 ## Issues
 None
 
 ## Contributing 
 
 You are well to pull request, for major changes, please open an issue. First discuss what you would like to change. please make sure to update tests as appropriate.
 
 ## Licence
 Copyright (C) 2020 David Gabriel
 
 
 
 ## References
 - MachineLearningmastery.com/time-series-forecasting-lstm-neural-networks-keras
 - [Yahoo Finance](https://query1.finance.yahoo.com/v7/finance/download/Fperiod1=156289894&period2=1595312294&interval=1d&event=history,)
 - [TensorFlow](https://www.tensorflow.org/guide/keras/rnn)
 - [Towardsdatascince](http://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470)
 - Tong et al., 2019 W. Tong, L. Li, X. Zhou, A. Hamilton, K. Zhang Deep Learnong PM2 6 concentration with bidirectional LSTM RNN
 
