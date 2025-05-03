🔍 Multi-Level Sentiment Analysis
This project presents a tiered approach to sentiment analysis, offering three progressively more powerful models — from lightweight binary classification to fine-tuned transformer-based emotion detection.

📊 **Overview**

| Model    | Dataset      | Type                         | Class Count | Use Case                                             | Accuracy | Computational Cost |
|----------|--------------|------------------------------|--------------|------------------------------------------------------|----------|---------------------|
| Model 1  | Sentiment140 | Binary (Positive/Negative)   | 2            | Real-time or low-resource applications               | ~79%     | 🟢 Low              |
| Model 2  | GoEmotions   | Multi-class (Emotions)       | 27           | Moderate applications with richer emotion detection  | ~43%     | 🟡 Low to Medium    |
| Model 3  | GoEmotions   | Multi-class (Emotions)       | 27           | High-accuracy, nuanced sentiment analysis            | ~92%     | 🔴 High             |


NOTE: Ready out-of-the-box fine-tuned BERT sentiment analysis have an accuracy of 88%-93% depending if it's for binary or multiclass classification (Depends on number of classes as well) 

🧠 Model Breakdown


🔹 Model 1: Logistic Regression (Binary Classification)
Dataset: Sentiment140

Task: Classifies sentiment as either Positive or Negative.

Model: TF-IDF vectorization + Logistic Regression

Use Case: Best for scenarios with:

Low latency requirements

Limited computational power (e.g., mobile, edge devices)

Real-time filtering or monitoring

Accuracy: ~77% (baseline)

Limitations: Cannot distinguish nuanced emotions or sarcasm, negation handling is a bit weak.



🔸 Model 2: XGBoost (Emotion Classification)
Dataset: GoEmotions

Task: Classifies text into 27 emotion categories (e.g., happy, angry, surprised)

Model: TF-IDF + XGBoost classifier

Use Case: Ideal for:

Chatbots or virtual assistants

Emotion-aware product feedback analysis

Applications where training time is less critical than inference speed




🔻 Model 3: Fine-tuned BERT (Emotion Classification)
Dataset: GoEmotions

Task: Emotion classification using fine-tuned BERT

Use Case: Perfect for:

Research and high-stakes NLP applications

Products with backend GPU/TPU support

Psychological or sociological analysis of language

Pros:

Captures nuanced meanings, sarcasm, and context

State-of-the-art results on multi-class sentiment tasks

Cons:

High training/inference cost

Requires GPU for practical deployment





📌 Notes
TF-IDF vectorization is used in both Models 1 and 2.

All models follow a standard pipeline: Preprocessing → Vectorization → Training → Evaluation → Prediction

Preprocessing includes but is not limited to stop word removal, URL/mention stripping, punctuation cleanup, and lemmatization.

🤖 Future Enhancements
Add live web demo

Model ensemble to combine predictions

Sarcasm detection layer for better classification
