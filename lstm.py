
class Lstm():


  from math import sqrt
  from numpy import concatenate
  from matplotlib import pyplot
  import pandas as pd
  from datetime import datetime
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.preprocessing import LabelEncoder
  from sklearn.metrics import mean_squared_error
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM



  def __init__(self, my_dataset):
    self.dataset=my_dataset
    self.values=my_dataset.values
    pass


  def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
      data: Sequence of observations as a list or NumPy array.
      n_in: Number of lag observations as input (X).
      n_out: Number of observations as output (y).
      dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
      Pandas DataFrame of series framed for supervised learning.
    """  
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    self.agg=agg
    return agg

  # A coluna a ser predita deve ser a 1, ou dá errado
  def fit(self,lag=1, lead=1,column_to_predict=0):
    # integer encode direction
    tam=len(self.dataset.columns)
    encoder = LabelEncoder()
    # ensure all data is float
    values = self.values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = self.series_to_supervised(scaled, lag, lead) #n de timesteps atras e pra frente q vc quer ver
    # drop columns we don't want to predict
    """ #isso é feito apagando todos que vem dps de N_colunas_inicial+1
    # ou seja, se a BD tem 8 colunas inicialmente[0-7], antes de passar pelo series_to_supervised(), logo eu vou apagar a coluna 9 em diante (n apago a 8)
    #isso se a coluna a ser predita for a 1 coluna/col[0] da bd (sem contar o index)
    #por enquanto eu so posso escolher a coluna 0 para predizer. dps vou fazer um módulo para trocar a coluna de desejada de lugar com a coluna 0
    """
    reframed=reframed.iloc[:,:(tam+1)]
    self.reframed=reframed
    self.scaler=scaler
    

  def split_supervised(self,porcentagem_para_treinamento=70):
    # split into train and test sets
    values = self.reframed.values
    X = self.dataset.values
    n_train = int(len(X) * (porcentagem_para_treinamento/100))
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    self.train_X, self.train_y,self.test_X, self.test_y=train_X,train_y,test_X,test_y
    
  def reshape_3d(self):
    # reshape input to be 3D [samples, timesteps, features]
    train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
    test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
    self.train_X=train_X
    self.test_X=test_X

  def make_lstm(self,epochs=2,batch_size=30,out_dim=35,lag=1, lead=1, column_to_predict=0):
    """
    The firstlayer/dimension has X neurons,X being the batch_size
    if batch_size=1/'None' it's then equal to the input_shape/how many examples you give for training
    vc tem 100 exemplos, mas o batch é 30, ele so pega 30 exs por iteração, se vc n especifica, ele pega 100
    """
    self.fit(lag, lead, column_to_predict)
    self.split_supervised()
    self.reshape_3d()
    model = Sequential()
    model.add(LSTM(out_dim, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(self.train_X, self.train_y, epochs=epochs, batch_size=batch_size, validation_data=(self.test_X, self.test_y), verbose=1, shuffle=False)
    self.history=history
    self.model=model
    self.lead=lead
    
  def plot_history(self):
    pyplot.clf()
    pyplot.plot(self.history.history['loss'], label='train')
    pyplot.plot(self.history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    
  def predict(self,x=0):
    yhat = self.model.predict(self.test_X)
    test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[2]))
    self.yhat=yhat
    self.test_X=test_X
    #yhat é a predição de test_X
    points=self.yhat[:self.lead]
    self.points=points
    if x==0:
      pyplot.clf()
      pyplot.plot(points,'bo')
      pyplot.show()

  def rmse(self):
    # invert scaling for forecast
    inv_yhat = concatenate((self.yhat, self.test_X[:, 1:]), axis=1)
    inv_yhat = self.scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = self.test_y.reshape((len(self.test_y), 1))
    inv_y = concatenate((test_y, self.test_X[:, 1:]), axis=1)
    inv_y = self.scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    casos=inv_yhat[:self.lead]
    if casos<0:
      casos=0
    else:
      casos=int(casos)
    print('\n\n Test RMSE: %.3f \n\n' % rmse,'\n casos novos=',casos)
    self.rmse=rmse
    
  def statistics(self):
    self.plot_history()
    self.rmse()
    
  def run_all(self,epochs=2,batch_size=100,out_dim=30,lag=1, lead=1, column_to_predict=0):
    #lead da o numero de pontos do grafico, quantos dias na frente
    self.make_lstm(epochs,batch_size,out_dim,lag, lead,column_to_predict)
    self.predict()
    self.statistics()
    return(self.points)