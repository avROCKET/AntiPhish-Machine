import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# Ensure necessary NLTK downloads
nltk.download('omw-1.4')
nltk.download('wordnet')

# Define helper functions for data augmentation
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_words = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_words)
    num_replaced = 0
    for random_word in random_words:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence

# Load dataset
data = pd.read_csv('.\\assets\\phishing_data_by_type_updated - LSTM.csv', encoding='ISO-8859-1')
texts = data['Text'].fillna("Missing").astype(str)
labels = data['Type']

# Split data to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Apply data augmentation only on the training set
augmented_texts = [synonym_replacement(text, n=1) for text in X_train]
X_train_augmented = X_train.tolist() + augmented_texts
y_train_augmented = y_train.tolist() + y_train.tolist()

# Initialize and fit the tokenizer on the augmented training data
tokenizer = Tokenizer(num_words=7500, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_augmented)

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train_augmented)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=200, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=200, padding='post', truncating='post')

# Encode the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_augmented)
y_test_encoded = encoder.transform(y_test)

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=7500, output_dim=64, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train_encoded, epochs=7, validation_data=(X_test_padded, y_test_encoded))

# Save the model
model.save('phishing_detection_model_4.h5')

# Save the tokenizer to a file
tokenizer_json = tokenizer.to_json()
with open('tokenizer4.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
