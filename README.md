# Cyberbullying-Prediction-using-Machine-Learning

# Overview
This project implements a machine learning-based cyberbullying detection system that analyzes social media text data to classify messages as either bullying or non-bullying. The model leverages various supervised learning techniques, including Support Vector Machines (SVM), Random Forest, Logistic Regression, and Stochastic Gradient Descent (SGD), to achieve high accuracy in identifying cyberbullying instances.

# Features
1. Data Preprocessing:
- Tokenization, punctuation removal, stopword filtering, and lemmatization.
- Feature extraction using CountVectorizer.
2. Machine Learning Models:
- Logistic Regression, Random Forest, Support Vector Machines (SVM), and SGDClassifier.
- Implemented a Voting Classifier for ensemble learning.
3. Web Application:
- Built using Flask, allowing users to input text and get real-time predictions.
- Signup and login functionality using an SQLite database.
4. Performance Metrics:
- Evaluated models using accuracy, precision, recall, F1-score, ROC-AUC score, and confusion matrix.


# Tech Stack
1. **Programming Languages:** Python, HTML, CSS
2. **Machine Learning Libraries:** Scikit-Learn, NLTK, NumPy, Pandas
3. **Web Framework:** Flask
4. **Database:** SQLite
5. **Visualization Tools:** Matplotlib, Seaborn

# How to Use:
- Clone the repository.
- To see the python code and machine learning model, open the notebook.ipynb file.
- To run the application, watch the video Execution.mp4 and follow the steps.

# Usage
- Sign up/Login: Users can create an account or log in.
- Text Classification: Enter a message to check if it contains cyberbullying content.

# Future Enhancements
- Implement deep learning models (LSTMs, BERT) for better accuracy.
- Integrate real-time monitoring for detecting cyberbullying on live platforms.
- Extend dataset coverage for multilingual and multimodal detection.
