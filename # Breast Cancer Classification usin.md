# Breast Cancer Classification using Support Vector Machine (SVM)

This project is a simple implementation of an SVM (Support Vector Machine) model to predict whether a tumor is malignant or benign. It uses the Breast Cancer Wisconsin dataset from Scikit-learn. 

## 📁 Project Structure

```
.
├── svm_cancer_classification.py   # Main Python script with model implementation
├── README.md                      # Project documentation
```

## 📊 Dataset

The dataset used is `load_breast_cancer()` from Scikit-learn, which includes 569 samples with 30 features each (such as mean radius, mean texture, etc.). Each sample is labeled as either:

- `0` = Malignant (cancerous)
- `1` = Benign (non-cancerous)

## 🔧 Technologies & Libraries

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 🚀 How to Run

1. Clone the repository or download the `svm_cancer_classification.py` file.
2. Make sure the required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the Python script:
   ```bash
   python svm_cancer_classification.py
   ```

## ⚙️ Workflow Summary

1. **Load Dataset** – Load the built-in breast cancer dataset from Scikit-learn.
2. **Preprocess** – Create a DataFrame and explore the features.
3. **Train-Test Split** – Split the data into training and testing parts (70%-30%).
4. **Train Model** – Use an SVM model (`SVC`) to train on the data.
5. **Initial Evaluation** – Check the confusion matrix and classification report to see how well the model works.
6. **Optimize with GridSearchCV** – Try different values for parameters like `C`, `gamma`, and `kernel` to find the best combination.
7. **Final Evaluation** – Evaluate the tuned model again.

## 📈 Evaluation Metrics

The model is evaluated using:

- Confusion Matrix
- Precision, Recall, F1-Score

These help check how accurately the model predicts tumor types.

## 📌 Notes

- Scaling the data (e.g. using StandardScaler) might improve performance.
- GridSearchCV can take a few minutes depending on your computer.

## ✅ Example Output (after tuning)

```
Best Parameters:
{'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}

Final Model Performance after GridSearch:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        64
           1       0.99      0.99      0.99       107

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171
```

## 📬 Contact

I'm always open to feedback or suggestions. Feel free to open an issue if you have ideas or find bugs.

---

Made with ❤️ as part of my learning journey using Scikit-learn.

