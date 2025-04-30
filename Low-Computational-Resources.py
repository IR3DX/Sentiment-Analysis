#Train Model:-------------------------------------------------------------------
#Libraries:---------------------------------------------------------------------
import os
import kagglehub
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Initialize lemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Function to lemmatize words
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Handle negations by combining the negation with the word to be vectorized together
def handle_negations(text):
    negations = r"\b(not|no|never|n't|isn't|wasn't|can't|won't|doesn't)\b"
    text = re.sub(negations + r"\s+(\w+)", r"not_\1", text, flags=re.IGNORECASE)  
    return text


#Main code:---------------------------------------------------------------------


# Download DataSet and define the path
path = kagglehub.dataset_download("kazanova/sentiment140")

# Add the file name to the path
file_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")

# Load into a pandas DataFrame
df = pd.read_csv(file_path, encoding="ISO-8859-1", header=None)

# Assign column names
df.columns = ["sentiment", "id", "date", "query", "user", "text"]

# Drop irrelevant data
df = df[["sentiment", "text"]]

# Convert sentiment 4 to 1 (0 = negative, 1 = positive) [Label Encoding] {Binary Classification}
df.loc[:, "sentiment"] = df["sentiment"].replace(4, 1)

# Data Pre-processing and cleaning
df['processed_text'] = df['text'].str.lower() # Convert tweet text to lowercase
df['processed_text'] = df['processed_text'].apply(remove_urls) # Remove URLS
df['processed_text'] = df['processed_text'].apply(remove_mentions_hashtags) # Remove mentions and Hashtags
df['processed_text'] = df['processed_text'].apply(remove_special_chars) # Remove everything and keep letters and numbers (Remove punctitation, emojis, special chaeracters like $%&)
df['processed_text'] = df['processed_text'].apply(remove_stopwords) # Remove stopwords
df['processed_text'] = df['processed_text'].apply(lemmatize_text) # Lemmatize words
df['processed_text'] = df['processed_text'].apply(handle_negations) # Handle negations

# Vectorizing the text
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
vectorized_data = vectorizer.fit_transform(df['processed_text'])

# Data Split (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    vectorized_data,
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
    )

# Create and train the model
model = LogisticRegression(max_iter=1000, C=1)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Testing code:------------------------------------------------------------------


# Show first 5 rows to and full text of first row
#print(df['processed_text'][0])
#df.head()

# Vectorized DF
#vectorized_data_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())
#vectorized_data_df.head()

# Check the distribution of sentiment values
#df['sentiment'].value_counts()

# Trying a few values for C
#for C_value in [0.01, 0.1, 1, 10, 100]:
    #model = LogisticRegression(max_iter=1000, C=C_value)
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #print(f"Accuracy for C={C_value}: {accuracy_score(y_test, y_pred)}")


#Download the model:------------------------------------------------------------------
import joblib

# Downloading the Model
joblib.dump(model, 'sentiment_model.joblib')

# Downloading TF-IDF Vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and Vectorizer have been saved successfully!")


#Run the model:------------------------------------------------------------------
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading the model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Setting up function parameters
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing input function
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+|\S+\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]|_", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# User input
input_text = input("Please input a sentence to analyze the sentiment: ")

# Prprocessing input
processed_input = preprocess_input(input_text)
vectorized_input = vectorizer.transform([processed_input])

# Probabilty processing
probabilities = model.predict_proba(vectorized_input)[0]
positive_prob = probabilities[1]
negative_prob = probabilities[0]

# Analysis results
if positive_prob >= 0.85:
        sentiment = f"Very Positive ({positive_prob:.2f})"
elif positive_prob >= 0.65:
        sentiment = f"Positive ({positive_prob:.2f})"
elif negative_prob >= 0.85:
        sentiment = f"Very Negative ({negative_prob:.2f})"
elif negative_prob >= 0.65:
        sentiment = f"Negative ({negative_prob:.2f})"
else:
        sentiment = f"Neutral (Positive: {positive_prob:.2f}, Negative: {negative_prob:.2f})"

print("Sentiment prediction:", sentiment)


