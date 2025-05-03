#Train Model:-------------------------------------------------------------------
#Libraries:---------------------------------------------------------------------
!pip install -q kaggle xgboost

!mkdir -p data/full_dataset/
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv


import os
import kagglehub
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

#Functions:---------------------------------------------------------------------

# Function to Remove URLs with HTTP, www., and domains without www
def remove_urls(text):
    return re.sub(r"http\S+|www.\S+|\S+\.\S+", "", text)

# Function to Remove mentions and hashtags
def remove_mentions_hashtags(text):
    return re.sub(r"@\w+|#\w+", "", text)

# Function to remove special characters
def remove_special_chars(text):
    return re.sub(r"[^\w\s]|_", "", text)

# Download stop word list
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to remove stop words
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])



# Function to lemmatize words
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Handle negations by combining the negation with the word to be vectorized together
def handle_negations(text):
    negations = r"\b(not|no|never|n't|isn't|wasn't|can't|won't|doesn't)\b"
    text = re.sub(negations + r"\s+(\w+)", r"not_\1", text, flags=re.IGNORECASE)
    return text


#Main code:---------------------------------------------------------------------

# Load the three CSV files
df1 = pd.read_csv('data/full_dataset/goemotions_1.csv')
df2 = pd.read_csv('data/full_dataset/goemotions_2.csv')
df3 = pd.read_csv('data/full_dataset/goemotions_3.csv')

# Combine the three DataFrames into one
df = pd.concat([df1, df2, df3], ignore_index=True)

# Determine the dominant emotion for each row (the emotion with the highest value)
emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                   'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                   'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                   'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Create a new column 'dominant_emotion' that holds the name of the dominant emotion for each row
df['dominant_emotion'] = df[emotion_columns].idxmax(axis=1)



# Initialize lemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Data Pre-processing and cleaning
df['processed_text'] = df['text'].str.lower() # Convert text to lowercase
df['processed_text'] = df['processed_text'].apply(remove_special_chars) # Remove special characters
df['processed_text'] = df['processed_text'].apply(remove_stopwords) # Remove stopwords
df['processed_text'] = df['processed_text'].apply(lemmatize_text) # Lemmatize words
df['processed_text'] = df['processed_text'].apply(handle_negations) # Handle negations



# Encode target labels (dominant emotions)
label_encoder = LabelEncoder()
df['encoded_emotion'] = label_encoder.fit_transform(df['dominant_emotion'])


# Vectorizing the text
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
vectorized_data = vectorizer.fit_transform(df['processed_text'])


# Data Split (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    vectorized_data,
    df["encoded_emotion"],
    test_size=0.2,
    random_state=42,
    stratify=df["encoded_emotion"]
)

# Create and train the model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Testing code
# Show the first few rows to verify
#print(df.head())
#print(df.columns)
# Display the first few rows of the DataFrame to verify
#print(df[['text', 'dominant_emotion', 'encoded_emotion']].head())

#Download the model:------------------------------------------------------------------

# Downloading the Model
joblib.dump(model, 'xgboost_emotion_model.joblib')


# Save label encoder
joblib.dump(label_encoder, "label_encoder.joblib")

# Save the vectorizer
joblib.dump(vectorizer, "vectorizer.joblib")

print("Model and Vectorizer have been saved successfully!")

#Run the model:------------------------------------------------------------------
# Load required components
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading components
model = joblib.load('xgboost_emotion_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Preprocessing setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing input function
def preprocess_input(text):
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = handle_negations(text)
    return text

# User input
input_text = input("Please input a sentence to analyze the emotion: ")

# Processing and prediction
processed_input = preprocess_input(input_text)
vectorized_input = vectorizer.transform([processed_input])
probabilities = model.predict_proba(vectorized_input)[0]
predicted_index = probabilities.argmax()
predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
confidence = probabilities[predicted_index]

print(f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})")


