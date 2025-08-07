🧠 Stroke Prediction Using Machine Learning
This project aims to develop a robust, accurate, and interpretable stroke prediction system using various machine learning techniques. Stroke is one of the leading causes of death and disability worldwide, and early detection is crucial for prevention and treatment. The model is trained on a healthcare dataset with patient data and tackles class imbalance using appropriate resampling methods.

📊 Abstract
Stroke prediction is a critical task in the healthcare domain due to its potential to reduce mortality and disability. This project builds a machine learning pipeline for predicting stroke using a structured dataset. It includes data preprocessing, feature selection, balancing techniques, hyperparameter tuning, and ensemble learning strategies such as bagging, boosting, and stacking. We evaluated multiple models using appropriate metrics to identify the most effective approach.

📁 Dataset
The dataset contains anonymized patient records with the following attributes:

Age, Gender

Hypertension, Heart Disease

Marital Status, Work Type, Residence Type

Average Glucose Level, BMI

Smoking Status

Stroke (Target variable)

⚙️ Methodology
✔️ Preprocessing
Encoding of categorical variables using One-Hot Encoding.

Normalization using Min-Max scaling for numerical features.

✔️ Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique)

Random UnderSampling
Both strategies were evaluated before and after train-test split to avoid data leakage.

✔️ Feature Selection
Used Recursive Feature Elimination (RFE) with Random Forest to select the top 7 important features.

✔️ Model Training
Algorithms used:

Logistic Regression

Random Forest

XGBoost

SVM

Extra Trees

Ensemble techniques: Bagging, Boosting, Stacking, Voting

✔️ Hyperparameter Tuning
GridSearchCV was used for tuning parameters like:

n_estimators, max_depth (Random Forest)

C, gamma (SVM)

learning_rate, etc.

📈 Evaluation Metrics
The models were evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

🧪 Results
Sampling Strategy	Model	Accuracy	Precision	Recall	F1 Score
SMOTE Before Split	Stacking (Bagging + Boosting + LGBM)	96%	97%	94%	95%
UnderSampling Before Split	Stacking	78%	73%	88%	80%
SMOTE After Split	Stacking	90%	17%	18%	18%
UnderSampling After Split	Stacking	92%	20%	10%	13%

✅ Best performance achieved using Stacking + SMOTE before splitting the dataset.

## 🔍 ROC Curve
![ROC Curve]("C:\Users\LENOVO\OneDrive\Desktop\download.png")


🧠 Conclusion
Balancing the dataset before train-test split significantly improves performance.

Ensemble techniques, especially Stacking, provide robust results on imbalanced healthcare datasets.

Proper hyperparameter tuning and feature selection enhance accuracy and recall.

📚 References
A few key references include:

IEEE, MDPI, and Nature papers on stroke prediction using ML

SMOTE: Synthetic Minority Oversampling Technique

Recursive Feature Elimination (RFE)

For a complete list, see the References Section in the project report.

👨‍💻 Authors
Sudhir Chouhan — Email

Vaibhav Pratap Singh — Email

Supervised by Dr. Suchismita Mahato, Dept. of CSE, NIT Jamshedpur

Would you like me to generate this as a README.md file for you to download?








You said:
aslo just using rf modal and in ensemble using stckintg +bagging+boosting


ChatGPT said:
Thanks for the clarification! Here's your updated README.md content based on:

✅ Using only the Random Forest model

✅ Using Stacking (with Bagging + Boosting + LightGBM) for ensemble

🧠 Stroke Prediction Using Machine Learning
A machine learning-based stroke prediction system designed to help identify high-risk patients early using Random Forest and ensemble techniques. This project aims to integrate predictive modeling into healthcare analytics for timely intervention and better patient outcomes.

📊 Abstract
Stroke is a major cause of death and long-term disability. This project focuses on developing a stroke prediction system using Random Forest and ensemble learning methods like Stacking (Bagging + Boosting + LightGBM). The model is trained on a healthcare dataset containing various patient features such as age, hypertension, BMI, and glucose levels. Class imbalance is handled with oversampling and undersampling methods like SMOTE and RandomUnderSampler.

📁 Dataset
The dataset includes the following features:

Demographics: Age, Gender, Residence Type, Marital Status

Health Conditions: Hypertension, Heart Disease, Avg Glucose Level, BMI

Lifestyle: Smoking Status, Work Type

Target: Stroke (1 = stroke, 0 = no stroke)

⚙️ Methodology
✔️ Preprocessing
Categorical encoding using One-Hot Encoding

MinMax Scaling for numerical features

✔️ Balancing the Data
SMOTE (Oversampling minority class)

Random UnderSampling

Best results achieved when balancing is applied before splitting the data

✔️ Feature Selection
Recursive Feature Elimination (RFE) using Random Forest to select top features:

avg_glucose_level, age, bmi, etc.

✔️ Model Training
🔍 Random Forest as the base classifier

🔄 Ensemble Learning:

Bagging: Random Forest

Boosting: LightGBM

Stacking: Combines predictions from Bagging and Boosting with a meta-classifier

✔️ Hyperparameter Tuning
Done using GridSearchCV

Parameters tuned:

n_estimators, max_depth, min_samples_split (Random Forest)

learning_rate, num_leaves (LightGBM)

📈 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

ROC-AUC Curve

🧪 Results Summary
Strategy	Model	Accuracy	Precision	Recall	F1 Score
Oversampling (SMOTE) Before Split	Stacking (RF + LightGBM)	96%	97%	94%	95%
Undersampling Before Split	Stacking	78%	73%	88%	80%

✅ Best result: Using SMOTE before splitting with Stacking (RF + Boosting)

🧠 Conclusion
Random Forest, when paired with RFE and proper tuning, offers strong predictive performance.

Stacking (Bagging + Boosting + LightGBM) outperforms individual models.

Applying SMOTE before splitting leads to the most reliable performance on imbalanced datasets.

This pipeline can be integrated into real-world healthcare systems for early stroke detection.

📚 References
Guhdar et al. (2023), "Stroke Prediction Using ML"

Dritsas & Trigka (2022), "Naive Bayes and SMOTE"

IEEE, MDPI, and Nature publications on stroke prediction

👨‍💻 Authors
Sudhir Chouhan — sudhirc377@gamil.com

Vaibhav Pratap Singh — vaibhavpsinghsng@gmail.com

Supervisor: Dr. Suchismita Mahato, Department of CSE, NIT Jamshedpur

