import pickle
import random
import json
from flask import Flask, render_template, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load model and vectorizer
try:
    with open('chatbot_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

try:
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    print("✅ Vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load vectorizer: {e}")

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Clean input text
def clean_text(text):
    return text.lower()

# Predict intent
def predict_intent(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction

# Get response
def get_response(intent_name):
    for intent in intents["intents"]:
        if intent["tag"] == intent_name:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chatbot_response():
    user_text = request.args.get('msg')
    if user_text:
        intent = predict_intent(user_text)
        response = get_response(intent)
        return jsonify({"response": response})
    return jsonify({"response": "I didn't get that. Could you say it again?"})

if __name__ == "__main__":
    app.run(debug=True)
