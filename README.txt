# 🔍 Fake Review Detector

A machine learning web app that detects whether a product review is **genuine or fake** using NLP and a Linear SVM classifier.

## 🚀 Live Demo
[Click here to try it](#) <!-- replace with your Streamlit URL after deployment -->

## 📊 Model Performance
| Model | Accuracy |
|---|---|
| Logistic Regression | 89.9% |
| Naive Bayes | 87.6% |
| **Linear SVM** | **90.4%** ✅ |

## 🛠️ Tech Stack
- Python, scikit-learn, NLTK
- TF-IDF Vectorization (unigrams + bigrams)
- Streamlit for UI
- Trained on Amazon Fake Reviews Dataset (Kaggle)

## ⚙️ How It Works
1. User pastes a product review
2. Text is cleaned — lowercased, stopwords removed, lemmatized
3. TF-IDF converts text to numerical features
4. Linear SVM predicts Genuine or Fake with a confidence score

## ⚠️ Known Limitation
Training data consists of computer-generated fake reviews. Human-written promotional fakes may occasionally be misclassified.

## 📁 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📬 Contact
Built by Bilal Lodhi — www.linkedin.com/in/melodhi
