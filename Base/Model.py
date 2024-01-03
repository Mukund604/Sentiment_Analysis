import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle as pickle

df = pd.read_csv('C:\\Users\\avaada\\Documents\\GitHub\\Pavagada-Forecst\\Sentiment_Analysis\\Data\\tripadvisor_hotel_reviews.csv')
print(df.columns)


df = df[['Review', 'Rating']]
df['Sentiment' ] = df['Rating'].apply(lambda x : 'Positive' if x > 3 else 'Negative' if x < 3 else 'Nuetral')
df = df[['Review', 'Sentiment']]
df


tokenizer  = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Review'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df['Review'])
padded_seq = pad_sequences(sequences,maxlen=100,truncating='post')


sentimet_labels = pd.get_dummies(df['Sentiment']).astype(int).values
sentimet_labels     


X_train, X_test, y_train, y_test = train_test_split(padded_seq, sentimet_labels, test_size=0.2)
X_train,y_train


model = Sequential()
model.add(Embedding(5000, 100, input_length=100))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


y_pred = np.argmax(model.predict(X_test), axis=-1)
print("Accuracy:", accuracy_score(np.argmax(y_test, axis=-1), y_pred))

model.save('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)





