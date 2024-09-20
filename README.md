# 📱 SMS Spam Detection Project

This project is designed to classify SMS messages as spam or ham (not spam) using machine learning techniques. By leveraging natural language processing (NLP) for feature extraction, various classifiers are trained to achieve high prediction accuracy.

# 🗂 Table of Contents
Project Overview
Technologies Used
Dataset Information
Installation and Setup
Data Processing
Modeling Approach
Evaluation and Results
Usage Guide
Conclusion and Future Work

# 📌 Project Overview
The objective of this project is to detect spam SMS messages based on their content. The process involves:

Preprocessing SMS text using NLP techniques

Applying machine learning models like SVM, Naive Bayes, and Random Forest

Implementing ensemble techniques to improve classification accuracy

# 🔧 Technologies Used
Python: Core programming language

Scikit-learn: For model training and evaluation

NLTK: For natural language processing

Pandas & NumPy: Data handling and manipulation

TfidfVectorizer: Feature extraction using Term Frequency-Inverse Document Frequency

Matplotlib & Seaborn: Data visualization libraries

# 📊 Dataset Information
The dataset contains SMS messages labeled as spam or ham.

Each SMS message serves as the input feature, while the label (spam or ham) is the target variable.

Text processing includes cleaning, tokenization, and feature transformation using TF-IDF.

# 🧹 Data Processing
Text Preprocessing:
Lowercasing, punctuation removal, and stopword removal using NLTK.

Tokenization to break SMS text into words.

Feature Extraction:

Converting SMS text into numerical data using TF-IDF vectorization.

Train-Test Split:

Dividing the dataset into training and testing sets to evaluate model performance.

# 🤖 Modeling Approach
Multiple machine learning models were trained for spam classification:

Multinomial Naive Bayes (MNB): Well-suited for text classification tasks.

Support Vector Machine (SVM): Effective for high-dimensional data like text.

Random Forest Classifier: An ensemble method for improved accuracy.

Ensemble Methods:

Voting Classifier: Combines the predictions of multiple models for final predictions.

Stacking Classifier: Combines several models (e.g., SVM, Naive Bayes, Random Forest) into a meta-classifier for better prediction accuracy.


# 📊 Evaluation and Results
Models were evaluated using the following metrics:

Accuracy: The overall correctness of predictions.

Precision: The proportion of correctly predicted spam messages out of all predicted spam messages.

Recall: The proportion of actual spam messages correctly predicted by the model.

F1-Score: The harmonic mean of precision and recall, providing a balanced measure.

Performance Metrics:
Voting Classifier:

Accuracy: 98.16%
Precision: 99.22%
Stacking Classifier:

Accuracy: 98.06%
Precision: 94.96%
These results indicate high accuracy and precision, making the model suitable for detecting spam SMS messages.


# 🚀 Usage Guide
To classify new SMS messages using the trained model:

Run the Jupyter notebook SMS_SPAM.ipynb.

Follow the steps to load and preprocess the dataset, train the models, and test the predictions.

To classify a custom SMS message:

message = ["Congratulations! You've won a free lottery ticket."]
prediction = model.predict(vectorizer.transform(message))
print("Spam" if prediction == 1 else "Ham")


# 🔮 Conclusion and Future Work
This project demonstrates the effectiveness of machine learning techniques in spam detection. By using a combination of text preprocessing, TF-IDF for feature extraction, and powerful classifiers, the model achieves high accuracy.

Future Improvements:
Experimenting with deep learning techniques such as LSTMs for better context understanding.

Integrating more advanced NLP techniques (e.g., word embeddings like Word2Vec or BERT).

Further optimizing the ensemble models to boost precision and recall.

