# Install and Import Libraries--------------------------------------------
!pip install -q transformers datasets gradio scikit-learn

import torch
import numpy as np
import pandas as pd
import gradio as gr
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Functions:--------------------------------------------
# Convert multi-labels to binary vectors
def encode_labels(example):
    label_vector = [0.0] * num_labels
    for label in example['labels']:
        label_vector[label] = 1.0
    example['label'] = label_vector
    return example

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",  
        truncation=True,
        max_length=128 
    )
# Define Metrics
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
    }

# Model training:--------------------------------------------
# Load and Preprocess Dataset
dataset = load_dataset("go_emotions")
label_names = dataset['train'].features['labels'].feature.names
num_labels = len(label_names)
dataset = dataset.map(encode_labels)


# Tokenization
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir="./goemotions_model",
    do_train=True,
    do_eval=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs'
)

tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
print(tokenized_dataset["train"][0])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train Model
trainer.train()

# Save Model:----------------------------------------------------------------------

model.save_pretrained("goemotions_model")
tokenizer.save_pretrained("goemotions_model")

# Gradio Interface:---------------------------------------------------------------
label_names = dataset['train'].features['labels'].feature.names
model = AutoModelForSequenceClassification.from_pretrained("goemotions_model")
tokenizer = AutoTokenizer.from_pretrained("goemotions_model")

def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).numpy()[0]
    return {label: float(f"{prob:.2f}") for label, prob in zip(label_names, probs) if prob > 0.3}

gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence..."),
    outputs=gr.Label(num_top_classes=10),
    title="GoEmotions BERT Classifier",
    description="Multi-label emotion classification from Google's GoEmotions dataset."
).launch()
