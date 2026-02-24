import torch, json
import pickle
import numpy as np
from torch import nn
from flask import Flask, request, jsonify


model = nn.Linear(1000, 1)
model.load_state_dict(torch.load("sentiment.pt"))


model.eval()

cv = pickle.load(open("count_vecotirzer.pkl", "rb"))

app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "You need to send a POST request with the text data to predict."
    if request.method == "POST":
        data = json.loads(request.data.decode("utf-8"))
        print(data)
        text = data["text"]
        x = cv.transform([text])
        x = torch.tensor(x.todense(), dtype=torch.float32)
        y = nn.functional.sigmoid(model(x))
        result = "Spam" if y[0].item() > 0.01 else "Ham"
        return jsonify({result: y[0].item()})

if __name__ == "__main__":
    app.run(debug=True)