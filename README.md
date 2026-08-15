# EDP-week-3
Absolutely. Below is a **complete README.md** specifically for your YouTube Comment Spam Detection project, using the results you actually obtained rather than generic numbers.

# YouTube Comment Spam Detection Using NLP

## 1. Problem Statement

YouTube receives a large number of comments every day, including genuine user discussions as well as promotional, misleading, and unwanted spam comments. Manually identifying spam comments is difficult and time-consuming.

The objective of this project is to develop a **machine learning-based YouTube comment spam detection system** that automatically classifies comments into two categories:

* **Spam**
* **Not Spam**

The project uses **Natural Language Processing (NLP)** techniques to convert textual comments into numerical features and applies **Multinomial Naive Bayes** for classification.

The main workflow is:

```text
YouTube Comments
       ↓
Data Cleaning
       ↓
Text Preprocessing
       ↓
Train/Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Naive Bayes Classification
       ↓
Spam / Not Spam
       ↓
Model Evaluation
```

---

## 2. Dataset

The project uses the **YouTube Comments Dataset with Sentiment, Toxicity and Spam Labels**.

The original dataset contains:

* **45,005 comments**
* **14 columns**
* YouTube-related metadata
* Comment text
* Sentiment labels
* Toxicity labels
* Spam labels

Important columns include:

```text
comment_text
label_spam
label_sentiment
label_toxicity
```

For this project, the primary columns used are:

```text
comment_text → Input feature
label_spam   → Target variable
```

The sentiment and toxicity labels were not used for the primary spam classification task because the objective of this project is specifically **spam detection**.

---

## 3. Dataset Analysis

Before building the machine learning model, Exploratory Data Analysis (EDA) was performed to understand the dataset.

### Dataset Information

The dataset initially contained:

```text
Rows: 45,005
Columns: 14
```

The `comment_text` and `label_spam` columns contained no missing values.

Some other columns contained missing values, including:

* `video_title`
* `author`
* `language`

Since the spam detection model only requires the comment text and spam label, these missing values did not affect the primary classification task.

### Spam Distribution

The dataset was highly imbalanced:

```text
Not Spam: 44,189
Spam:        816
```

Percentage distribution:

```text
Not Spam: 98.19%
Spam:      1.81%
```

This imbalance became an important consideration during model evaluation.

### Comment Length Analysis

Comment length was also analyzed to understand the distribution of text.

A boxplot showed that there were many long-comment outliers, particularly among non-spam comments. These comments were retained because a long comment is not necessarily spam.

---

## 4. Data Cleaning

Data cleaning was performed before applying NLP techniques.

The following operations were performed:

### Duplicate Removal

The dataset contained:

```text
5,909 duplicate comments
```

Duplicate comments were removed to reduce repetition and prevent the same text from having excessive influence during training.

### Missing Text and Labels

The primary columns were checked for missing values:

```text
comment_text → No missing values
label_spam   → No missing values
```

### Original and Cleaned Text

The original comment was preserved in:

```text
comment_text
```

and the processed version was stored in:

```text
clean_text
```

This allowed the original and processed text to be compared during analysis.

---

## 5. Text Preprocessing

Text preprocessing was performed to convert raw YouTube comments into cleaner text suitable for machine learning.

The following preprocessing techniques were studied and/or applied:

### Lowercasing

All text was converted to lowercase.

Example:

```text
"THIS VIDEO IS AMAZING"
```

became:

```text
"this video is amazing"
```

### URL Removal

URLs were removed because they can introduce unnecessary variation in the text.

Example:

```text
"Check https://example.com now"
```

became approximately:

```text
"check now"
```

### Punctuation and Special Character Removal

Unnecessary punctuation and special characters were removed.

Example:

```text
"Great video!!!"
```

became:

```text
"great video"
```

### Extra Space Removal

Multiple spaces were normalized into single spaces.

### Tokenization

Tokenization was studied using NLTK to understand how sentences are divided into individual tokens.

Example:

```text
"This YouTube video is amazing!"
```

becomes:

```text
['This', 'YouTube', 'video', 'is', 'amazing', '!']
```

### Stopwords

Stopwords such as common English words were studied and tested using NLTK.

For the baseline spam classifier, aggressive stopword removal was not used because common words can sometimes contribute to text classification patterns.

---

## 6. TF-IDF Feature Extraction

Machine learning algorithms cannot directly process raw text. Therefore, the cleaned comments were converted into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF assigns importance to words based on their frequency in a document and their frequency across the collection of documents.

The project used:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

### Configuration

```text
Maximum features: 5,000
N-grams: Unigrams + Bigrams
```

### Unigrams

Individual words:

```text
free
money
subscribe
channel
```

### Bigrams

Two-word combinations:

```text
free money
subscribe channel
check channel
```

Using both unigrams and bigrams allows the model to learn individual words as well as short phrases.

### Train/Test TF-IDF

The dataset was divided into training and testing sets before TF-IDF was fitted.

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

`fit_transform()` was used only on the training data, while `transform()` was used on the test data to avoid data leakage.

The resulting feature matrices were:

```text
Training: (24,660, 5,000)
Testing:   (6,166, 5,000)
```

---

## 7. Naive Bayes

The primary machine learning algorithm used in this project is **Multinomial Naive Bayes**.

Multinomial Naive Bayes is commonly used for text classification because it works effectively with word-frequency and TF-IDF-based features.

The basic workflow is:

```text
Cleaned Text
      ↓
TF-IDF
      ↓
Numerical Features
      ↓
Multinomial Naive Bayes
      ↓
Spam / Not Spam
```

The baseline model was created using:

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
```

---

## 8. Class Imbalance

One of the most important findings in this project was the severe class imbalance.

The original dataset contained approximately:

```text
98.19% Not Spam
1.81% Spam
```

Because of this imbalance, accuracy alone is not a reliable measure of model performance.

For example, a model that predicts almost everything as "not spam" can achieve very high accuracy while still failing to identify actual spam.

Therefore, this project evaluates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Special attention was given to **spam recall**, because missing an actual spam comment is an important error in a spam detection system.

---

## 9. Model Experiments

Multiple Naive Bayes configurations were tested to understand the effect of class imbalance.

### Experiment 1 — Multinomial Naive Bayes Baseline

The baseline model produced:

```text
Accuracy: 99.16%
```

Spam performance:

```text
Precision: 100%
Recall:     30%
F1-score:   46%
```

Confusion matrix:

```text
[[6092, 0],
 [  52, 22]]
```

The model produced **0 false positives**, but it missed **52 actual spam comments**.

This demonstrated that the baseline model was very conservative when predicting spam.

---

### Experiment 2 — Complement Naive Bayes

Complement Naive Bayes was tested as an additional experiment.

Confusion matrix:

```text
[[5627, 465],
 [  32, 42]]
```

Approximate results:

```text
Accuracy:       91.94%
Spam Precision:  8.28%
Spam Recall:    56.76%
Spam F1-score:  14.49%
```

This model detected more spam than the baseline, increasing spam recall from approximately **30% to 57%**.

However, it also produced **465 false positives**, meaning many legitimate comments were incorrectly classified as spam.

Therefore, Complement Naive Bayes was not selected as the final model.

---

### Experiment 3 — Balanced Multinomial Naive Bayes

A modified Multinomial Naive Bayes model was created using adjusted class priors:

```python
model_balanced = MultinomialNB(
    class_prior=[0.90, 0.10]
)
```

This gave the spam class more importance during classification.

Results:

```text
Accuracy: 99.29%
Spam Precision: 92%
Spam Recall:    45%
Spam F1-score: 60%
```

Confusion matrix:

```text
[[6089, 3],
 [  41, 33]]
```

This model reduced the number of missed spam comments from **52 to 41** while producing only **3 false positives**.

---

## 10. Evaluation

The models were evaluated using multiple metrics.

### Accuracy

Measures the percentage of all predictions that were correct.

The final model achieved:

```text
99.29% accuracy
```

### Precision

Measures how many comments predicted as spam were actually spam.

Final spam precision:

```text
92%
```

This means that most comments classified as spam were actually spam.

### Recall

Measures how many actual spam comments were successfully detected.

Final spam recall:

```text
45%
```

The baseline model had approximately 30% spam recall, so the balanced model improved spam detection.

### F1-score

F1-score combines precision and recall.

Final spam F1-score:

```text
60%
```

This improved from the baseline F1-score of approximately 46%.

### Confusion Matrix

The final model produced:

```text
                 Predicted
              Not Spam   Spam

Actual
Not Spam        6089       3
Spam              41      33
```

Therefore:

```text
True Negative  = 6089
False Positive = 3
False Negative = 41
True Positive  = 33
```

---

## 11. Error Analysis

Error analysis was performed to understand why the classifier made incorrect predictions.

For the baseline model, several spam comments were classified as non-spam.

Examples included comments related to:

* Promotional offers
* Download links
* Scholarships
* Free trials
* Product advertisements
* Channel promotions

This showed that some spam comments can look similar to normal conversational comments.

The baseline model missed:

```text
52 spam comments
```

The balanced Multinomial Naive Bayes model reduced this to:

```text
41 spam comments
```

However, the model still does not detect every spam comment.

This indicates that further improvements could be explored in future work, such as:

* Better text preprocessing
* Hyperparameter tuning
* Different TF-IDF configurations
* Class-weighting strategies
* More advanced machine learning models
* Word and character n-grams
* Deep learning or transformer-based NLP models

---

## 12. Final Results

The three tested models can be summarized as follows:

| Model                                |   Accuracy | Spam Precision | Spam Recall | Spam F1 |
| ------------------------------------ | ---------: | -------------: | ----------: | ------: |
| Multinomial Naive Bayes              |     99.16% |           100% |         30% |     46% |
| Complement Naive Bayes               |     91.94% |          8.28% |      56.76% |  14.49% |
| **Balanced Multinomial Naive Bayes** | **99.29%** |        **92%** |     **45%** | **60%** |

### Selected Model

The **Balanced Multinomial Naive Bayes model** was selected as the final model.

Although Complement Naive Bayes achieved higher spam recall, it generated a very large number of false positives.

The balanced Multinomial Naive Bayes model provided a better overall balance:

```text
Accuracy       → 99.29%
Spam Precision → 92%
Spam Recall    → 45%
Spam F1        → 60%
False Positive → 3
```

Therefore, it provides a more practical balance between detecting spam and avoiding incorrect classification of legitimate comments.

---

## 13. Conclusion

This project developed a YouTube comment spam detection system using Natural Language Processing and machine learning.

The complete pipeline included:

```text
Data Collection
      ↓
Data Exploration
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Train/Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Multinomial Naive Bayes
      ↓
Model Evaluation
      ↓
Class Imbalance Handling
      ↓
Model Comparison
```

The baseline Multinomial Naive Bayes model achieved **99.16% accuracy**, but its spam recall was only approximately **30%** because of the severe class imbalance.

To improve spam detection, Complement Naive Bayes and a class-prior-adjusted Multinomial Naive Bayes model were also tested.

The final Balanced Multinomial Naive Bayes model achieved:

```text
99.29% Accuracy
92% Spam Precision
45% Spam Recall
60% Spam F1-score
```

The project demonstrates that **accuracy alone is not sufficient for evaluating an imbalanced spam classification problem**. Precision, recall, F1-score, and confusion matrix analysis provide a more meaningful understanding of model performance.

Overall, the project successfully demonstrates the use of **NLP, text preprocessing, TF-IDF, Naive Bayes, class imbalance analysis, model comparison, and error analysis** for real-world YouTube comment spam detection.

