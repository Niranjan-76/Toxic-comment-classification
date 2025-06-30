
# Toxic Comment Classification 🧠🗨️

This project is a **binary text classification model** that identifies whether an input comment is **Toxic** or **Not Toxic**, using a **Bi-directional LSTM model** built with Keras.

---

## 📌 Dataset

- **Source:** [Kaggle - Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)
- Only two columns used: `comment_text` and `toxic`
- Labels: `1` = Toxic, `0` = Not Toxic

---

## 🧠 Model Architecture

- Tokenization with top 10,000 frequent words
- Sequence padding to max length 150
- Embedding layer (128 dimensions)
- Bi-directional LSTM (64 units)
- Output layer with sigmoid activation

---

## ✅ Features

- Allows users to **manually input** a comment and get real-time prediction.
- Users can also **upload a CSV** file with a `comment_text` column to classify multiple comments in bulk.
- Generates output file: `toxic_comment_predictions.csv`

---

## 📊 Performance

- **Training Accuracy:** 96.8%
- **Validation Accuracy:** 96.2%

---

## 📁 Files

| File                  | Description                                   |
|-----------------------|-----------------------------------------------|
| `toxic_classifier.py` | Main training and prediction script           |
| `README.md`           | Project documentation                         |
| `train.csv`           | Dataset file (from Kaggle - not included)     |

---

## 🚀 How to Run

1. Install required libraries:
```bash
pip install pandas numpy tensorflow
```

2. Run the Python script:
```bash
python toxic_classifier.py
```

3. Enter a comment OR upload a CSV file to classify.

---

## 📜 License

This project is for educational use. Dataset belongs to [Kaggle](https://www.kaggle.com/datasets/julian3833).

---

## 👤 Author

**Niranjan S**  
B.E. in AI & ML, Mysore University School of Engineering
