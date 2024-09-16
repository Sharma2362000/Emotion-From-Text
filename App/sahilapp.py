import tkinter as tk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and punkt (tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
model_name = "mahadev23/dissertation_sahil_bert"  #  repository name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Define label mapping
label_mapping = {0: 'joy', 1: 'sadness', 2: 'fear', 3: 'anger', 4: 'surprise', 5: 'neutral'}

def preprocess_sentence(sentence):
    # Truncate sentence to 300 characters
    sentence = sentence[:300]

    # Replace HTML line breaks with space
    sentence = re.sub(r"<br\s*/?>", " ", sentence)

    # Replace non-alphabetic characters (except apostrophes) with space
    sentence = re.sub(r"[^a-zA-Z']", " ", sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Tokenize (split by spaces)
    tokens = sentence.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    cleaned_sentence = ' '.join(tokens)

    return cleaned_sentence

def predict_single_string(text, tokenizer, model, device):
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()

    # Convert logits to predicted label
    predicted_label = np.argmax(logits, axis=1)

    return label_mapping[predicted_label[0]]

def on_predict():
    user_input = entry.get()
    cleaned_input = preprocess_sentence(user_input)
    result = predict_single_string(cleaned_input, tokenizer, model, device)
    result_textbox.delete(1.0, tk.END)
    result_textbox.insert(tk.END, f"Predicted Emotion: {result}")

# Tkinter UI setup
root = tk.Tk()
root.title("Emotion Prediction App")

# Label and Entry for input text
tk.Label(root, text="Enter Text:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

# Button to predict
predict_button = tk.Button(root, text="Predict Emotion", command=on_predict)
predict_button.pack(pady=10)

# Text box to display the result
result_textbox = tk.Text(root, height=5, width=50)
result_textbox.pack(pady=5)

# Start Tkinter main loop
root.mainloop()
