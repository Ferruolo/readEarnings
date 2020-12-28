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

if(tf.test.is_gpu_available()):
    print("GPU in use")
else:
    raise Exception("No GPU in use")  


print("Loading w2v file (takes forever")
w2v = gensim.models.KeyedVectors.load_word2vec_format("w2vfile", binary=True)


print("Loading finnhub")
fh_client = finnhub.Client(api_key = "bv2o3vn48v6ubfuli7h0")
WORD_DEPTH = 300
MAX_LEN = 3000 #Found in seperate program

#Define Model
#Train
try:
    model = tf.required_space_to_batch_paddings
except FileExistsError:
    model = Sequential()

    model.add(layers.Conv1D(4, 100, activation='relu', input_shape = (MAX_LEN, WORD_DEPTH)))

    model.add(layers.MaxPool1D())

    model.add(layers.Dropout(0.5))

    #Not in original model, but I need 1 d output

    model.add(layers.Flatten())

    model.add(layers.Dense(100))


    model.add(layers.Dense(1))

    model.build()

    model.compile(optimizer=keras.optimizers.Adadelta(), loss=losses.MeanSquaredError())

print("Model built, training model. See ya in a few years")



symbols = [equity['symbol'] for equity in  fh_client.stock_symbols('US')]
print("Train")
#Create Data
for symbol in symbols:
    try: 
        fin_data = fh_client.financials(symbol, 'ic', freq = 'quarterly')['financials'][:39]
        eps_estimates = fh_client.company_eps_estimates(symbol, freq = 'quarterly')['data'][9:(39+8)]
        # rev_estimates = fh_client.company_revenue_estimates(symbol, freq = 'quarterly')['data'][9:(39+8)]


        transcript_list = fh_client.transcripts_list(symbol)
        if len(transcript_list) > 1:



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
                for i in range(5, len(transcripts)):
                    eps_surprise = (fin_data[i]['dilutedEPS'] - eps_estimates[i]['epsAvg'])/eps_estimates[i]['epsAvg']
                    t_seq = transcript_text[i-5:i-1]

                    x = [np.concatenate(i[0], axis=0) for i in t_seq]

                    y = [np.array(i[1]) for i in eps_surprise]

                    max_len = max(x, key = lambda i: i.shape[0]).shape[0]
                    x_pad = list()
                    for i in x:
                        
                        difference = MAX_LEN - i.shape[0]
                        if difference != MAX_LEN:
                            pad_matrix = np.zeros((difference, 300))
                            
                            padded = np.concatenate([i, pad_matrix], axis=0)
                            x_pad.append(padded)
                        else:
                            x_pad.append(i)




                    x = np.stack(x_pad, axis=0)
                    y = np.stack(y, axis=0)
                    





                    batch_size = len(stock)
                    


                    model.fit(x=x, y=y, batch_size = batch_size, epochs=500, verbose=2)
                    model.save('model.h5')
    finally:
        print(symbol)


