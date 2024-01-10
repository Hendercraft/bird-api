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
        if 'files' not in request.files:
            return jsonify({'error': 'No file part'})

        files = request.files.getlist('files')
        results = []

        for file in files:
            if file.filename == '':
                results.append({'error': 'No selected file'})
                continue

            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((224, 224))  # Resize the image
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)[0]

            # Process each file similar to before
            top_pred_index = np.argmax(prediction)
            all_predictions = [(i, prob) for i, prob in enumerate(prediction)]
            all_predictions.sort(key=lambda x: x[1], reverse=True)

            image_result = {
                'top_prediction': {
                    'label': labels_dict[top_pred_index],
                    'probability': round(prediction[top_pred_index] * 100, 3)
                },
                'other_predictions': []
            }

            sum_small_probs = 0
            for index, prob in all_predictions:
                if index != top_pred_index:
                    if prob >= 0.0001:
                        image_result['other_predictions'].append({
                            'label': labels_dict[index],
                            'probability': round(prob * 100, 3)
                        })
                    else:
                        sum_small_probs += prob

            if sum_small_probs > 0:
                image_result['other_predictions'].append({
                    'label': 'Other',
                    'probability': round(sum_small_probs * 100, 3)
                })

            results.append(image_result)

        return jsonify(results)
    return render_template('upload.html')