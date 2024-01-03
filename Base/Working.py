import keras
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import customtkinter
from CTkMessagebox import CTkMessagebox
from textblob import TextBlob

model = keras.models.load_model('C:\\Users\\avaada\\Documents\\GitHub\\Pavagada-Forecst\\Sentiment_Analysis\\Base\\Saved_Model\\sentiment_analysis_model.h5')
with open('C:\\Users\\avaada\\Documents\\GitHub\\Pavagada-Forecst\\Sentiment_Analysis\\Base\\Saved_Model\\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def runAnalysis():
    text = inputText.get()
    predict_sentiment(text)


def predict_sentiment(text):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)[0]
    if np.argmax(predicted_rating) == 0:
        # return 'Negative'
        CTkMessagebox(message="Negative.",icon="warning")
    elif np.argmax(predicted_rating) == 1:
        # return 'Neutral'
        CTkMessagebox(message="Nuetral.",icon="check")
    else:
        CTkMessagebox(message="Positive.",icon="check")


root = customtkinter.CTk()
root.geometry(f"{500}x{300}")
root.title("Sentiment Analysis")
root.configure(bg='black')

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

label = customtkinter.CTkLabel(master = root, text="Enter a sentence:", font=('Montserrat', 17))
label.pack(pady = 30)

inputText = customtkinter.CTkEntry(master = root, placeholder_text="Enter your sentence", width=150, justify = 'center')
inputText.pack()

analyze_button = customtkinter.CTkButton(master=root,text="Analyze Sentiment", command=runAnalysis,width=40,  corner_radius=20, font=('Montserrat', 17))
analyze_button.pack(pady=30)


root.mainloop()