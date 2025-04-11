import os
import cv2
import numpy as np
import tensorflow as tf
import shap
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import pickle

# Configuration
UPLOAD_FOLDER = 'static/uploads'
EXPLANATION_FOLDER = 'static/explanations'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (168, 168)

# Translations
TRANSLATIONS = {
    'en': {
        'title': 'Brain Tumor Classifier',
        'upload_text': 'Upload MRI Image',
        'result_title': 'Classification Result',
        'tumor_detected': 'Tumor Detected',
        'no_tumor_detected': 'No Tumor Detected',
        'confidence': 'Confidence',
        'classes': 'Possible Classes',
        'explanation': 'Model Explanation'
    },
    'zh': {
        'title': '脑肿瘤分类器',
        'upload_text': '上传 MRI 图像',
        'result_title': '分类结果',
        'tumor_detected': '检测到肿瘤',
        'no_tumor_detected': '未检测到肿瘤',
        'confidence': '置信度',
        'classes': '可能的分类',
        'explanation': '模型解释'
    }
}

# Class mappings
CLASS_MAPPINGS = {
    0: 'Glioma', 
    1: 'Meningioma', 
    2: 'No Tumor', 
    3: 'Pituitary'
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPLANATION_FOLDER'] = EXPLANATION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload and explanation directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

# Load trained model
try:
    model = load_model('model.keras')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_loaded = False

# Initialize SHAP explainer
explainer = None

def initialize_explainer():
    """Initialize the SHAP explainer with proper class names"""
    global explainer
    if model is not None and explainer is None:
        try:
            # Define prediction function
            def f(x):
                return model(x)
            
            # Define class labels from our mapping
            class_labels = list(CLASS_MAPPINGS.values())
            
            # Create the masker with the correct shape
            masker_blur = shap.maskers.Image("blur(168,168)", (168, 168, 1))
            
            # Create the SHAP explainer with proper output names (class labels)
            explainer = shap.Explainer(f, masker_blur, output_names=class_labels)
            return True
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            return False
    return explainer is not None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction using TensorFlow"""
    try:
        # Step 1: Read image file
        image_string = tf.io.read_file(image_path)
        
        # Step 2: Decode image
        image = tf.image.decode_jpeg(image_string, channels=3)  # Decode as 3-channel (RGB)
        
        # Step 3: Resize image
        image_dim = (168, 168)
        image = tf.image.resize(image, image_dim)
        
        # Step 4: Convert to grayscale
        image = tf.image.rgb_to_grayscale(image)
        
        # Step 5: Normalize pixel values
        image = image / 255.0
        
        # Step 6: Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # Convert to NumPy for compatibility with existing code
        processed_img = image.numpy()
        
        # Read original image for potential display purposes
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        return processed_img, original_img
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None, None

def analyze_image(image_path):
    """Analyze image and return classification results"""
    if not model_loaded:
        return {
            'success': False,
            'error': 'Model not loaded'
        }
    
    try:
        # Preprocess image
        processed_img, original_img = preprocess_image(image_path)
        
        if processed_img is None:
            return {
                'success': False,
                'error': 'Image preprocessing failed'
            }
        
        # Predict
        predictions = model.predict(processed_img)[0]
        
        # Get best prediction
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_MAPPINGS[predicted_class_index]
        confidence = predictions[predicted_class_index] * 100
        
        # Detailed probabilities
        probabilities = {
            cls: float(prob * 100) for cls, prob in 
            zip(CLASS_MAPPINGS.values(), predictions)
        }
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_class_index': int(predicted_class_index),
            'confidence': float(confidence),
            'probabilities': probabilities
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_standard_shap_explanation(image_path, predicted_class_index):
    """Generate standard SHAP explanation for the image"""
    if not initialize_explainer():
        return None
    
    try:
        # Preprocess image for SHAP
        processed_img, _ = preprocess_image(image_path)
        
        if processed_img is None:
            return None
        
        # Generate unique filename for explanation
        filename = os.path.basename(image_path)
        explanation_filename = f"standard_explanation_{filename}"
        explanation_path = os.path.join(app.config['EXPLANATION_FOLDER'], explanation_filename)
        
        # Set matplotlib to not use GUI
        plt.ioff()  # Turn off interactive mode
        
        # Get class names
        class_labels = list(CLASS_MAPPINGS.values())
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Add a title
        plt.suptitle(f"Standard SHAP Explanation - Predicted Class: {class_labels[predicted_class_index]}", 
                    fontsize=16, fontweight='bold')
        
        # Generate SHAP values
        shap_values = explainer(processed_img, max_evals=1000, batch_size=50)

        # Create explanation plot
        shap.image_plot(
            [shap_values.values[:, :, :, :, i] for i in range(4)],
            processed_img,
            class_labels
        )
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                   "Red areas positively contribute to the class. Blue areas negatively contribute.",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Save the figure
        plt.savefig(explanation_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return explanation_filename
            
    except Exception as e:
        print(f"Error generating standard SHAP explanation: {e}")
        import traceback
        traceback.print_exc()
        return None



BACKGROUND_DATA_FILE = 'static/background_data.pkl'

def generate_deep_shap_explanation(image_path, predicted_class_index):
    """Generate DeepExplainer SHAP visualization"""
    global background_data
    
    if not model_loaded:
        return None
    
    try:
        # Preprocess image
        processed_img, _ = preprocess_image(image_path)
        
        if processed_img is None:
            return None
        
        # Generate unique filename
        filename = os.path.basename(image_path)
        explanation_filename = f"deep_explanation_{filename}"
        explanation_path = os.path.join(app.config['EXPLANATION_FOLDER'], explanation_filename)
        
        # Get class names
        class_labels = list(CLASS_MAPPINGS.values())
        
        # Set matplotlib to not use GUI
        plt.ioff()  # Turn off interactive mode
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Add title
        plt.suptitle(f"DeepExplainer SHAP Visualization - Predicted Class: {class_labels[predicted_class_index]}", 
                    fontsize=16, fontweight='bold')
        
        try:
            # Create the deep explainer with the background data
            deep_explainer = shap.DeepExplainer(model, background_data)
            deep_shap_values = deep_explainer.shap_values(processed_img)
            
            # Reshape the shap_values
            reshaped_shap_values = []
            for i in range(4):  # For each class
                reshaped_shap_values.append(deep_shap_values[0][:,:,:,i])

            shap.image_plot(reshaped_shap_values, processed_img[0], class_labels)

            # Add explanation text
            plt.figtext(0.5, 0.01, 
                      "Red areas positively contribute to the class. Blue areas negatively contribute.",
                      ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            # Save figure
            plt.savefig(explanation_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            return explanation_filename

        except Exception as e:
            print(f"DeepExplainer visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error generating DeepExplainer explanation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def load_background_data():
    """Load background data for SHAP explanations on app startup"""
    global background_data
    
    if os.path.exists(BACKGROUND_DATA_FILE):
        print("Loading background data from disk...")
        try:
            with open(BACKGROUND_DATA_FILE, 'rb') as f:
                background_data = pickle.load(f)
            print("Background data loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load background data: {e}")
            background_data = None
    return False




@app.route('/')
def home():
    """Home page route"""
    lang = session.get('lang', 'en')
    return render_template('index.html', 
                          lang=lang, 
                          translations=TRANSLATIONS[lang],
                          model_loaded=model_loaded)

@app.route('/set_language/<lang>')
def set_language(lang):
    """Set application language"""
    if lang in TRANSLATIONS:
        session['lang'] = lang
    return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    lang = session.get('lang', 'en')
    translations = TRANSLATIONS[lang]
    
    if not model_loaded:
        return jsonify({
            'success': False, 
            'error': 'Model not loaded. Please check server logs.'
        })
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image
        result = analyze_image(filepath)
        
        if result['success']:
            # Generate explanations if requested
            standard_explanation = None
            deep_explanation = None
            
            if request.form.get('generate_explanation') == 'true':
                standard_explanation = generate_standard_shap_explanation(
                    filepath, 
                    result['predicted_class_index']
                )
                
                deep_explanation = generate_deep_shap_explanation(
                    filepath, 
                    result['predicted_class_index']
                )
            
            return jsonify({
                'success': True,
                'filename': filename,
                'result': result,
                'standard_explanation': standard_explanation,
                'deep_explanation': deep_explanation
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Analysis failed')
            })
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/explanations/<filename>')
def get_explanation(filename):
    """Serve explanation images"""
    return send_file(os.path.join(app.config['EXPLANATION_FOLDER'], filename))

# Create a custom error handler for exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    print(f"Unhandled exception: {str(e)}")
    return jsonify({
        'success': False,
        'error': f"Server error: {str(e)}"
    }), 500

if __name__ == '__main__':
    load_background_data()
    port = int(os.environ.get("PORT", 8000)) # Default to 8000 if not set
    app.run(host="0.0.0.0", port=port)
    # Initialize explainer in background
    # Note: We'll initialize the explainer on demand instead of at startup
    #app.run(debug=False)  # Set debug to False to avoid issues with multiple threads