def SentimentAnalysis():
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
    print(df.head())

SentimentAnalysis()