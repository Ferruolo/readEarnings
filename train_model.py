import finnhub
import pickle 
import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, Sequential
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import sklearn
w2v = gensim.models.KeyedVectors.load_word2vec_format("w2vfile", binary=True)

fh_client = finnhub.Client(api_key = "bv2o3vn48v6ubfuli7h0")
WORD_DEPTH = 300

#Define Model
#Train


symbols = [equity['symbol'] for equity in  fh_client.stock_symbols('US')]

#Create Data
data = list()
for symbol in symbols:
    fin_data = fh_client.financials(symbol, 'ic', freq = 'quarterly')['financials'][:39]
    eps_estimates = fh_client.company_eps_estimates(symbol, freq = 'quarterly')['data'][9:(39+8)]
    # rev_estimates = fh_client.company_revenue_estimates(symbol, freq = 'quarterly')['data'][9:(39+8)]


    transcript_list = fh_client.transcripts_list(symbol)

    transcripts = [t for t in transcript_list['transcripts']]

    transcripts = [t['id'] for t in transcripts if 'Conference' not in t['title']]
    transcript_text = list()


    for t in transcripts[1:39]:
        raw = fh_client.transcripts(t)
        section = raw['transcript']
        text = [person['speech'] for person in section]
        text_str = ""
        for l in text:
            text_str += l[0]
            text_str += ' '



        tf_idf = TfidfVectorizer()
        vecs = tf_idf.fit_transform(text_str)
        tokens = tf_idf.inverse_transform(vecs)

        #Convert to w2v
        vectors = list()
        for call in tokens:
            w2v_call = list()
            for word in call:
                try:
                    w2v_call.append(w2v[word])
                except KeyError:
                    pass
            vectors.append(w2v_call)



        transcript_text.append(vectors)
    security_data = list()
    for i in range(5, len(transcripts)):
        eps_surprise = (fin_data[i]['dilutedEPS'] - eps_estimates[i]['epsAvg'])/eps_estimates[i]['epsAvg']
        t_seq = transcript_text[i-5:i-1]

        x = [np.concatenate(i[0], axis=0) for i in t_seq]

        y = [np.array(i[1]) for i in eps_surprise]

        max_len = max(x, key = lambda i: i.shape[0]).shape[0]

        #Padding
        x_pad = list()
        for i in x:
            
            difference = max_len - i.shape[0]
            if difference != 0:
                pad_matrix = np.zeros((difference, 300))
                
                i = np.concatenate([i, pad_matrix], axis=0)
                x_pad.append(i)
            else:
                x_pad.append(i)

        security_data.append({'transcript': , 
                            'eps_surprise': eps_surprise})
    data.append(security_data)


max_len = 0 
for stock in data:
    for sdata in stock
        for t in sdata['transcript']
            if t.shape[0] > max_len:
                max_len = t.shape[0]
    x_pad = list()
    for i in x:
        
        difference = max_len - i.shape[0]
        if difference != 0:
            pad_matrix = np.zeros((difference, 300))
            
            i = np.concatenate([i, pad_matrix], axis=0)
            x_pad.append(i)
        else:
            x_pad.append(i)

    test = x_pad.pop()
    test_y = y.pop()

    x = x_pad
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)





model = Sequential()

model.add(layers.Conv1D(4, 100, activation='relu', input_shape = (max_len, WORD_DEPTH)))

model.add(layers.MaxPool1D())

model.add(layers.Dropout(0.5))

#Not in original model, but I need 1 d output

model.add(layers.Flatten())

model.add(layers.Dense(100))


model.add(layers.Dense(1))

model.build()

model.compile(optimizer=keras.optimizers.Adadelta(), loss=losses.MeanSquaredError())



import math
test = data[(3*math.floor(len(data)/4)):]
data =  data[:(3*math.floor(len(data)/4))]

with open("test.pkl", "wb") as f:
    pickle.dump(f, test)    



for stock in data:
    batch_size = len(stock)

    x = [sdata['transcript'] for sdata in stock]
    y = [sdata['eps_surpise'] for sdata in stock]

    model.fit(x=x, y=y, batch_size = batch_size, epochs=500, verbose=2)
    model.save('model.h5')



