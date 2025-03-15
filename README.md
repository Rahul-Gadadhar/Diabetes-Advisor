AI-Powered Personalized Diabetes Nutrition Advisor using RNN  
================

AI-powered Diabetes Nutrition Advisor is a deep learning tool that integrates these components to assist diabetic patients in selecting foods that promote better glycemic control. By preprocessing patient and nutritional data, training an RNN model, and leveraging Google’s Gemini Generative AI for concise nutritional explanations, the system offers a comprehensive approach to dietary management for diabetes.  

## Table of Contents  
  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Datasets](#datasets)  
- [Usage](#usage)
- [Outputs](#Outputs)  
- [Model Architecture](#model-architecture)  
- [Dependencies](#dependencies)  
  
Overview
--------
The Diabetes Advisor is a Python-based tool designed to assist diabetic patients in selecting foods that can help manage their blood sugar levels.  
By combining diabetes patient data with detailed nutritional information of foods, the system preprocesses the data, trains an RNN model, and then uses user input (age, gender, diabetes type, blood sugar level) to generate personalized food recommendations.  
Additionally, the application utilizes Google’s Gemini Generative AI to provide short, structured explanations on how the recommended foods can aid in diabetic control.
  
Diabetes  
-----------------  
Diabetes is a chronic metabolic disorder characterized by high blood sugar levels due to the body’s inability to produce enough insulin or effectively use it. There are primarily two types: Type 1, an autoimmune condition where the pancreas produces little or no insulin, and Type 2, where the body becomes resistant to insulin. Proper glycemic control is essential to prevent complications like cardiovascular disease, nerve damage, and kidney issues, making regular monitoring and appropriate dietary management crucial.  

RNN  
-----------------  
A Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data by maintaining context through feedback loops. This makes RNNs particularly useful for analyzing time-series or sequential information, such as trends in nutritional data and patient health metrics. In this project, the RNN processes a combination of patient data and nutritional information to predict personalized food recommendations, thereby helping diabetic patients manage their blood sugar levels effectively.  
  
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
```bash
   git clone https://github.com/Rahul-Gadadhar/diabetes-advisor.git  
   cd diabetes-advisor
```
2. Create a Virtual Environment (Optional but Recommended):
```bash
   python -m venv venv  
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
3. Install Dependencies:

   Install the required Python packages using pip:
```bash
   pip install torch numpy pandas scikit-learn google-generativeai google-colab ipython
```
   Note: The "google-colab" package is required if you are running the code in a Google Colab environment.

Datasets
--------
This project requires two datasets:  
- Diabetes Patient Data: A CSV file (diabetes_patient_data.csv) containing patient information such as age, gender, blood sugar levels, and medication details.  
- Food Nutritional Data: A CSV file (pred_food.csv) with detailed nutritional information including glycemic index, calories, macronutrients, and micronutrients.

Place these CSV files in your Google Drive or local file system.  
If using Google Colab, update the file paths accordingly. For example:  

```python
diabetes_data = pd.read_csv(r'/content/drive/My Drive/Datasets/diabetes_patient_data.csv')
pred_food_data = pd.read_csv(r'/content/drive/My Drive/Datasets/pred_food.csv')
```


Usage
-----
1. Mount Google Drive (if using Colab):

   When running in Google Colab, use the following code snippet to mount your Google Drive:
```python
   from google.colab import drive  
   drive.mount('/content/drive')
```
2. Run the Script:

   Execute the Python script:
```python
   python diabetes_advisor.py
```
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

Outputs
------------------
User Inputs:  
![image](https://github.com/user-attachments/assets/3ff0771e-daff-41e6-ad43-71f27a6b0021)  
  
Generated Recommendations:  
![image](https://github.com/user-attachments/assets/c1b983ba-50ac-4d0c-bd29-57f48e30a216)  

Accuracy Metrics:  
![image](https://github.com/user-attachments/assets/ffe227a8-ab05-4288-9e81-b8215e579968)  

Gemini Response:  
![image](https://github.com/user-attachments/assets/51743f56-7435-49b9-b431-49ed717b9617)  
    
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

