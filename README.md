Real-Time Credit Card Fraud Detection Using Machine Learning

🎖️ Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions in real-time. By leveraging various supervised learning techniques, including ensemble models and anomaly detection methods, this system can effectively identify fraudulent behavior, which is critical for minimizing financial losses in the credit card industry.

📊 Dataset

Source: Kaggle - Credit Card Fraud Detection Dataset [http://kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download]

Records: 284,807 transactions

Features:
30 anonymized features (V1, V2, ..., V28)

Note on Features:
The dataset includes 28 anonymized PCA features (V1–V28) preprocessed via dimensionality reduction. These features are used directly without scaling, while only ‘Time’ and ‘Amount’ are scaled to normalize their range. This ensures the model leverages the full signal contained in these transformed components for effective fraud detection.

Time: Time of transaction in seconds since the first transaction
Amount: Amount of the transaction
Class: Target variable, 1 for fraud and 0 for legitimate transactions

This dataset is highly imbalanced with fraud transactions making up less than 1% of the total dataset, making it a good example for handling class imbalance.

🛠️ Technologies Used

Python Libraries: Pandas, NumPy

Scikit-Learn (Logistic Regression, Random Forest, XGBoost)

Imbalanced-learn (SMOTE)

SHAP (for model interpretability)

Matplotlib, Seaborn (for visualizations)

🛠️ Development Tools:

Jupyter Notebooks for experimentation

Git for version control

GitHub for project collaboration

📈 Key Steps

1. Exploratory Data Analysis (EDA)
Visualized distribution of fraudulent vs non-fraudulent transactions.
Analyzed key features, transaction amounts, and time-related patterns.

2. Data Preprocessing
Handled missing values, removed outliers, and scaled numerical features.
Encoded categorical variables (if necessary).

3. Class Imbalance Handling
Used SMOTE (Synthetic Minority Over-sampling Technique) and undersampling methods to balance the dataset.

4. Model Building
Built and evaluated models including Logistic Regression, Random Forest, and XGBoost.
Fine-tuned hyperparameters to achieve optimal performance.

5. Model Evaluation
Evaluated models using ROC-AUC, F1-Score, Precision-Recall, and Confusion Matrix.
Focused on maximizing recall to ensure frauds are detected early.

6. Model Explainability
Used SHAP to interpret model predictions and understand which features contributed most to fraud detection.

🏆 Results

ROC-AUC Score: 0.98 (indicating excellent model performance).

Recall: 95% (minimized false negatives, critical for fraud detection).

Precision: 98% (ensuring that flagged transactions are likely fraud).
The model demonstrates high accuracy in detecting fraud, ensuring that it is reliable for use in real-world financial systems.


🧹 Future Improvements

Implement real-time fraud detection with streaming data (e.g., Kafka).

Deploy the trained model as an API using FastAPI or Flask for real-time use.

Experiment with deep learning models such as Autoencoders for anomaly detection.

Test the model on additional datasets to improve generalization.

⚡ How to Run the Project

1. Clone this repository:
   git clone https://github.com/Neha-Mathew-08/credit-card-fraud-detection.git

   cd credit-card-fraud-detection

3. Install the dependencies:
   pip install -r requirements.txt



