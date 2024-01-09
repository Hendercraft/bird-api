import io
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json

app = Flask(__name__)
# Load the JSON file
json_file_path = 'model/labels.json' 
with open(json_file_path, 'r') as file:
    labels_dict = json.load(file)
labels_dict = {int(k): v for k, v in labels_dict.items()}

# Load the Model
model = keras.models.load_model('model/bird_model.keras')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((224, 224))  # Resize the image
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)[0]

            # Find the index of the highest prediction
            top_pred_index = np.argmax(prediction)

            # Create a sorted list of all predictions
            all_predictions = [(i, prob) for i, prob in enumerate(prediction)]
            all_predictions.sort(key=lambda x: x[1], reverse=True)

            # Prepare the response dictionary
            response = {
                'top_prediction': {
                    'label': labels_dict[top_pred_index],
                    'probability': round(prediction[top_pred_index] * 100, 3)
                },
                'other_predictions': []
            }

            # Initialize the sum of small probabilities
            sum_small_probs = 0

            for index, prob in all_predictions:
                if index != top_pred_index:
                    if prob >= 0.0001:
                        response['other_predictions'].append({
                            'label': labels_dict[index],
                            'probability': round(prob * 100, 3)
                        })
                    else:
                        sum_small_probs += prob

            # Add the sum of small probabilities as 'Misc'
            if sum_small_probs > 0:
                response['other_predictions'].append({
                    'label': 'Misc',
                    'probability': round(sum_small_probs * 100, 3)
                })

            return jsonify(response)
    return render_template('upload.html')