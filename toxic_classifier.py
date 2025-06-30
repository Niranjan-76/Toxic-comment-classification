
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load and preprocess training data
df = pd.read_csv("train.csv")[['comment_text', 'toxic']]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment_text'])
X = tokenizer.texts_to_sequences(df['comment_text'])
X = pad_sequences(X, maxlen=150)
y = df['toxic'].values

# Define model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=150))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=512, validation_split=0.2)

# --- Test Section ---

def predict_comment(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=150)
    predictions = model.predict(padded)
    return ["Toxic" if p > 0.5 else "Not Toxic" for p in predictions]

# 1. Manual Input
manual_input = input("Enter a comment to check for toxicity: ")
print("Prediction:", predict_comment([manual_input])[0])

# 2. File Upload (CSV with 'comment_text' column)
file_input = input("Enter CSV filename to test (or press Enter to skip): ")
if file_input:
    test_df = pd.read_csv(file_input)
    comments = test_df['comment_text'].astype(str).tolist()
    results = predict_comment(comments)
    test_df['Prediction'] = results
    test_df.to_csv("toxic_comment_predictions.csv", index=False)
    print("Predictions saved to toxic_comment_predictions.csv")
