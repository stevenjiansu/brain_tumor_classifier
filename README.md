# MediExplain AI - Brain Tumor Classifier

MediExplain AI is an innovative tool designed to assist medical professionals in diagnosing brain tumors using MRI scans. It leverages advanced AI models to classify tumors into categories such as Glioma, Meningioma, Pituitary Tumor, or No Tumor. The tool also provides transparent explanations for its predictions using SHAP visualizations.

## Features

- **Brain Tumor Classification**: Classifies MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
- **Explainable AI**: Generates SHAP visualizations to explain the model's predictions.
- **Multi-language Support**: Supports English and Chinese for the user interface.
- **User-friendly Interface**: Easy-to-use web interface for uploading MRI scans and viewing results.
- **Clinical Recommendations**: Provides recommendations based on the classification results.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd brain_tumor_classifier_test
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following directories exist:
   - `static/uploads`: For storing uploaded MRI images.
   - `static/explanations`: For storing SHAP explanation images.

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:8000`.

3. Upload an MRI image in JPG, JPEG, or PNG format.

4. View the classification results, confidence scores, and SHAP visualizations.

5. Download a detailed report in PDF format if needed.

## File Structure

```
brain_tumor_classifier_test/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # HTML template for the web interface
├── static/
│   ├── css/
│   │   └── styles.css      # Custom CSS for styling
│   ├── js/
│   │   └── script.js       # Custom JavaScript for interactivity
│   ├── uploads/            # Directory for uploaded MRI images
│   └── explanations/       # Directory for SHAP explanation images
└── README.md               # Project documentation
```

## Environment Variables

- `PORT`: Port number for the Flask server (default: 8000).

## Troubleshooting

- **Model not loaded**: Ensure the `model.keras` file is present in the root directory.
- **SHAP explainer issues**: Verify that the background data file (`background_data.pkl`) exists and is correctly loaded.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [SHAP](https://github.com/slundberg/shap) for explainable AI visualizations.
- [TensorFlow](https://www.tensorflow.org/) for deep learning.
- [Flask](https://flask.palletsprojects.com/) for the web framework.