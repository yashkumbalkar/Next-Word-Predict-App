import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


from preprocess import preprocess_text,preprocess_lines

# Reading Text File
file_path = 'human_chat.txt'

with open(file_path, "r") as file:
  lines = file.readlines()


# Preprocessing
preprocessed_lines = preprocess_lines(lines)

# Tokenizing
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_lines)
unique_words = len(tokenizer.word_index)
word_to_idx = tokenizer.word_index

# Load the Model
model_path = 'lstm_model.keras'
model = load_model(model_path)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])



# Streamlit app
# Streamlit UI
st.set_page_config(page_title="Next Word Predictor", page_icon="⌨", layout="wide")
st.title("Next Word Prediction using LSTM")
st.write("Enter a sequence of words to predict the next word:")

st.markdown(
    """
    ### Welcome to the Next Word Predictor
    This application predicts the next word in a sentence based on the text you provide. Powered by a trained LSTM model, it can generate words based on your input.

    ### How to Use:
    1. Type a sentence or a few words.
    2. Click "Predict Next Word" to get the next word suggestion.
    """
)



# Predict the Next Word
def predict_next_word(model ,input_text, word_to_idx):
  input_text = preprocess_text(input_text)
  # tokenize
  token_text = tokenizer.texts_to_sequences([input_text])[0]
  # padding
  padded_token_text = pad_sequences([token_text], maxlen = 151, padding='pre')
  # predict
  prediction = model.predict(padded_token_text , verbose = 0) # To suppress the output of Keras during predictions
  pos = np.argmax(prediction)

  for word,index in word_to_idx.items():
    if index == pos:
      return word





# User input text box
input_text = st.text_area("Enter text", height=100, placeholder="Type your sentence here...")


# Predict button
if st.button("Predict Next Word"):
    if input_text:
        with st.spinner("Generating the next word..."):
            for i in range(5):
                predicted_word = predict_next_word(model, input_text, word_to_idx)
                st.write(f"Next word after '{input_text}': {predicted_word}")
                input_text = input_text + " " + predicted_word
            
    else:
        st.warning("Please enter a sentence to predict the next word.")



