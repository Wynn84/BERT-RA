from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load your fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('C:/Users/wynna/OneDrive/Documents/1BERT/fine_tuned_bert_model')
tokenizer = BertTokenizer.from_pretrained('C:/Users/wynna/OneDrive/Documents/1BERT/fine_tuned_bert_model')


# Define the prediction function
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits (raw output)
    logits = outputs.logits

    # Apply sigmoid to get probabilities for multi-label classification
    probabilities = torch.sigmoid(logits)

    # Convert probabilities to binary predictions using a threshold (0.3 in this case)
    threshold = 0.3
    predicted_labels = (probabilities >= threshold).int().squeeze().tolist()

    # Define your label classes (update this to match your actual model's classes)
    label_classes = ['R.A 7877', 'R.A 11313', 'R.A 6949', 'R.A 9710', 'R.A 6725', 'R.A 9262']

    # Map the predicted binary labels to the corresponding class names
    predicted_class_names = [label_classes[i] for i, pred in enumerate(predicted_labels) if pred == 1]

    return predicted_class_names

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_api():
    # Get the input data from the request (assumes input in JSON format)
    data = request.json
    text = data.get('text', '')

    # Get the prediction from the model
    prediction = predict(text)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
