import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string,re

def pre_process(i):
    i = re.compile(r'<[^>]+>').sub('',i)
    i = re.sub(r"\s+[a-zA-Z]\s+", ' ', i)
    i = re.sub(r"([0-9]+)", ' ', i)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    i = i.translate(table)
    i = re.sub(r'\s+', ' ', i)
    return i

def predict (review,model_type):
    review = pre_process(review)
    if model_type == 1:
        model = load_model('movie_one.h5')
    elif model_type == 2:
        model = load_model('movie_two.h5')
    else:
        model = load_model('movie_three.h5')
    
    with open('tokenizer.pickle','rb') as  handle:
        tokenizer = pickle.load(handle)
    
    ans = tokenizer.texts_to_sequences([review])
    ans = pad_sequences(ans, maxlen=500)

    predictions = model.predict(ans)[0][0]

    return predictions


st.title('Analyse Movie Review')

review = st.text_area("Enter Your Review!")

if st.button("Predict with LSTM "):
    with st.spinner('Predicting.....'):
        prediction = predict(review,1)

        if prediction >= 0.6:
            st.success('Positive Review!')
        elif prediction <= 0.4:
            st.error('Negative Review!')
        else:
            st.warning('Neutral, try again')
elif st.button("Predict with DNN "):
    with st.spinner('Predicting.....'):
        prediction = predict(review,2)

        if prediction >= 0.6:
            st.success('Positive Review!')
        elif prediction <= 0.4:
            st.error('Negative Review!')
        else:
            st.warning('Neutral, try again')
elif st.button("Predict with Conv1D "):
    with st.spinner('Predicting.....'):
        prediction = predict(review,3)

        if prediction >= 0.6:
            st.success('Positive Review!')
        elif prediction <= 0.4:
            st.error('Negative Review!')
        else:
            st.warning('Neutral, try again')