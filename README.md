# Fake News Detector 🕵️‍♂️

A comprehensive machine learning system designed to classify the truthfulness of public statements by combining NLP techniques with speaker metadata.

## 📖 Overview
Developed as part of a Computer Science curriculum, this project aims to combat global misinformation by looking beyond just text. The system analyzes not only the **statement** itself but also the **contextual metadata** (speaker history, party affiliation, and job title) to provide a more accurate truthfulness score.

## 📊 Dataset
The project utilizes the **LIAR dataset**, which includes 12,836 real-world statements verified by PolitiFact. 
Statements are classified into 6 labels of truthfulnesS:
* ✅ **True**
* ✔️ **Mostly True**
* 🟡 **Half True**
* ⚠️ **Barely True**
* ❌ **False**
* 🔥 **Pants on Fire**

## 🛠️ Features & Engineering
To improve accuracy, this project implements advanced **Feature Engineering**:
* **Truth Index:** A calculated reliability coefficient based on the speaker's historical record.
* **Lexical Diversity:** Measuring the richness of the vocabulary used in statements.
* **POS Tagging:** Counting nouns and adjectives to detect subjective or emotional styles.
* **NLP Pipeline:** Implements Stemming, Lemmatization, Stopword removal, and Count Vectorization.

## 🧠 Model Performance
[cite_start]Several models were tested to find the most effective classifier[cite: 189]:

| Model | Accuracy / F1 Score |
| :--- | :--- |
| **XGBoost** | **0.640** (Best) |
| **RandomForest** | 0.618 |
| Decision Tree | 0.483 |
| Naive Bayes | 0.268 |
| GRU / LSTM | ~0.220 |


## 💻 UI Interface
The project includes a functional web interface built with **Streamlit**. Users can enter:
1. The text of a statement.
2. The speaker's name and job title.
3. The speaker's previous credit history (number of past true/false statements).

The system then outputs a **Truthfulness Prediction** and a probability chART.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/v4ni0/liar-plus-fake-news-detector.git](https://github.com/v4ni0/liar-plus-fake-news-detector.git)

   . Launch the App
2. Run streamlit interface
   ```streamlit run app.py ```
