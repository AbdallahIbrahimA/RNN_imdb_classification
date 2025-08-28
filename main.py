# Step 1: Import Libraries and Load the Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, InputLayer
import numpy as np

# Custom InputLayer to handle batch_shape issue
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

@st.cache_resource
def create_and_load_model():
    # Rebuild the exact model architecture
    model = Sequential([
        FixedInputLayer(input_shape=(239,)),  # Use input_shape instead of batch_shape
        Embedding(input_dim=10000, output_dim=32, input_length=239),
        SimpleRNN(128, activation='relu'),
        Dense(1, activation="sigmoid")
    ])
    
    # Compile the model (same as original)
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    # Load weights from the .keras file
    try:
        # Method 1: Try to load weights directly from .keras file
        with h5py.File('model/simple_rnn_imdb.keras', 'r') as f:
            if 'model_weights' in f:
                # Load weights layer by layer
                for layer in model.layers:
                    if layer.name in f['model_weights']:
                        weight_values = []
                        for weight_name in f[f'model_weights/{layer.name}']:
                            weight_values.append(f[f'model_weights/{layer.name}/{weight_name}'][()])
                        if weight_values:
                            layer.set_weights(weight_values)
        
        st.success("✅ Model weights loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        return None

# Load model
model = create_and_load_model()

if model:
    st.success("Model ready for predictions!")
    # Rest of your app code...
else:
    st.error("Failed to load model")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review




# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
    print(sentiment , prediction)
else:
    st.write('Please enter a movie review.')









