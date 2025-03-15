## **Deployed App on Streamlit link :-** [click here](https://yashkumbalkar-next-word-predict-app-app-sbogii.streamlit.app/)

# Next Word Prediction using LSTM

### Overview :-

This project uses Long Short-Term Memory (LSTM) networks implemented in Keras to predict the next word in a sequence of text. The model 
is trained on a small corpus of text data, and the application is deployed as a web app using Streamlit for real-time interaction.

### Data Source :-

The dataset used for this project is sourced from Kaggle:- [Human Conversation Text](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)

### Project Description :-

The LSTM model is trained on a small corpus of text. You can modify the training data or the model's architecture for custom use cases. 
Here's a quick outline of the training process:

- `Preprocessing:` Tokenizing the text, padding sequences, and creating training data.
- `Model Architecture:` Using an LSTM layer followed by a dense layer with a softmax activation for multi-class classification.
- `Training:` The model is trained for several epochs, with the `categorical_crossentropy` loss function and `adam` optimizer.

### Features :-

- `Next Word Prediction`: Given a sequence of words, the model predicts the next most likely word.
- `Real-time Prediction`: Users can input text and get a real-time word prediction.
- `Interactive Web Interface`: The Streamlit app provides a simple, user-friendly interface for testing the model.

### Example Usage :-

- Input a sequence of words (e.g., "hello how are") into the input box.
- The model predicts the next word (e.g., "you").
- The prediction is displayed on the web app.



