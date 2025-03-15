Diabetes Advisor
================

A machine learning-powered diabetes advisor that recommends suitable foods for diabetic patients based on their nutritional content and current blood sugar levels.  
This project uses a Recurrent Neural Network (RNN) to predict food recommendations from a dataset of diabetes patient data and food nutritional values, and further leverages Google’s Generative AI (Gemini) to provide a concise explanation of the nutritional benefits for diabetic control.

Table of Contents
-----------------
- Overview
- Features
- Installation
- Datasets
- Usage
- Model Architecture
- Dependencies
- License
- Acknowledgements

Overview
--------
The Diabetes Advisor is a Python-based tool designed to assist diabetic patients in selecting foods that can help manage their blood sugar levels.  
By combining diabetes patient data with detailed nutritional information of foods, the system preprocesses the data, trains an RNN model, and then uses user input (age, gender, diabetes type, blood sugar level) to generate personalized food recommendations.  
Additionally, the application utilizes Google’s Gemini Generative AI to provide short, structured explanations on how the recommended foods can aid in diabetic control.

Features
--------
- Data Preprocessing:  
  - Label encoding for categorical variables (Gender, Type of Diabetes, Medications, etc.).  
  - Scaling and normalization of continuous features.

- Model Training:  
  - A simple RNN model built with PyTorch.  
  - Training loop with real-time metrics (loss, accuracy, precision, recall, and F1-score).  
  - Saving the trained model for future inference.

- Food Recommendation:  
  - Predicting food recommendations based on processed input data.  
  - Displaying a main recommendation table along with detailed nutritional information.

- AI Explanation:  
  - Integrating Google’s Generative AI (Gemini) to generate concise nutritional explanations.  
  - Dynamic feedback based on recommended foods’ nutritional profiles.

Installation
------------
1. Clone the Repository:  

   git clone https://github.com/yourusername/diabetes-advisor.git  
   cd diabetes-advisor

2. Create a Virtual Environment (Optional but Recommended):

   python -m venv venv  
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3. Install Dependencies:

   Install the required Python packages using pip:

   pip install torch numpy pandas scikit-learn google-generativeai google-colab ipython

   Note: The "google-colab" package is required if you are running the code in a Google Colab environment.

Datasets
--------
This project requires two datasets:  
- Diabetes Patient Data: A CSV file (diabetes_patient_data.csv) containing patient information such as age, gender, blood sugar levels, and medication details.  
- Food Nutritional Data: A CSV file (pred_food.csv) with detailed nutritional information including glycemic index, calories, macronutrients, and micronutrients.

Place these CSV files in your Google Drive or local file system.  
If using Google Colab, update the file paths accordingly. For example:  

   diabetes_data = pd.read_csv(r'/content/drive/My Drive/Datasets/diabetes_patient_data.csv')  
   pred_food_data = pd.read_csv(r'/content/drive/My Drive/Datasets/pred_food.csv')

Usage
-----
1. Mount Google Drive (if using Colab):

   When running in Google Colab, use the following code snippet to mount your Google Drive:

   from google.colab import drive  
   drive.mount('/content/drive')

2. Run the Script:

   Execute the Python script:

   python um_diabetes_advisor.py

3. Provide User Inputs:

   The script will prompt for:
   - Age  
   - Gender (Male/Female)  
   - Type of Diabetes (Type 1/Type 2)  
   - Number of food suggestions to display  
   - Blood sugar level (within the recommended range: 70-180 mg/dL)

4. Review the Recommendations:

   After training the model and processing the input data, the script will display:
   - A table of recommended foods along with key nutritional details.  
   - Accuracy metrics from the model training.  
   - An AI-generated explanation from Google’s Gemini on how the recommended foods can help with diabetic control.

Model Architecture
------------------
The system uses a basic RNN model implemented in PyTorch:
- Input Layer: Processes features such as age, blood sugar levels, glycemic index, calories, etc.  
- RNN Layer: Captures temporal patterns in the data.  
- Fully Connected Layer: Maps the RNN outputs to the prediction space corresponding to the number of food options.  
- Training: The model is trained using the Cross Entropy Loss and optimized with the Adam optimizer over 500 epochs with mini-batch training.

Dependencies
------------
- PyTorch – Deep learning framework used for building and training the RNN.  
- NumPy – Library for numerical computations.  
- Pandas – Data manipulation and analysis.  
- scikit-learn – Preprocessing, label encoding, scaling, and evaluation metrics.  
- Google Generative AI – Used for generating AI explanations.  
- Google Colab – (Optional) Environment for running the notebook.

License
-------
Distributed under the MIT License. See LICENSE for more information.

Acknowledgements
----------------
- Thanks to the contributors and the open-source community for the libraries and tools used in this project.  
- Special mention to Google for their Colab environment and Generative AI tools.
