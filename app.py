from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model_path = "model/constellation_model.h5"
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    try:
        image = image.resize((180, 180))  # Ensure image is resized properly
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

@app.route('/')
def index():
    return render_template('stars2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Open the image
            image = Image.open(file.stream)
            
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Get prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Print for debugging
            print("Prediction Probabilities:", prediction)
            print("Predicted Class Index:", predicted_class)

            # Define class names
            class_names = [
                'Andromeda', 'Antlia', 'Apus', 'Aquarius', 'Aquila', 'Ara', 'Aries', 'Auriga', 
                'Bootes', 'Caelum', 'Camelopardalis', 'Cancer', 'Canes Venatici', 'Canis Major', 
                'Capricornus', 'Carina', 'Cassiopeia', 'Centaurus', 'Chamaeleon', 'Columba', 
                'Corona Borealis', 'Corvus', 'Crater', 'Crux', 'Cygnus', 'Delphinus', 'Dorado', 
                'Draco', 'Equuleus', 'Eridanus', 'Fornax', 'Gemini', 'Grus', 'Hercules', 'Horologium', 
                'Hydra', 'Hydrus', 'Indus', 'Lacerta', 'Leo', 'Leo Minor', 'Lepus', 'Libra', 'Lupus', 
                'Lynx', 'Lyra', 'Malus', 'Mensa', 'Microscopium', 'Monoceros', 'Musca', 'Norma', 
                'Octans', 'Ophiuchus', 'Orion', 'Pavo', 'Pegasus', 'Perseus', 'Phoenix', 'Pictor', 
                'Piscis Austrinus', 'Pisces', 'Puppis', 'Pyxis', 'Reticulum', 'Sagitta', 'Sagittarius', 
                'Scorpius', 'Sculptor', 'Serpens', 'Southern Cross', 'Sculptor', 'Taurus', 'Triangulum', 
                'UrsaMajor', 'Vela', 'Virgo', 'Volans', 'Vulpecula'
            ]

            if predicted_class >= len(class_names):
                return jsonify({'error': 'Prediction out of bounds'}), 500

            result = class_names[predicted_class]

            return jsonify({'constellation': result})

        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
