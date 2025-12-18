# Predicting Customer Churn in Telecom

End‑to‑end DSML group project to predict customer churn for a telecom company using the IBM Telco Customer Churn dataset. The repository follows a `src/` + `notebooks/` structure so analysis is transparent, reproducible, and easy to extend.

## 1. Project Overview


Customer churn (customer attrition) is a key business problem for telecoms, where even a small increase in churn can significantly impact recurring revenue. 
This project builds machine learning models to identify customers at high risk of churning and highlights the drivers of churn so retention teams can take action. 

Main objectives:

- Clean and prepare the IBM Telco Customer Churn dataset for modeling.  
- Explore drivers of churn with EDA and feature engineering.  
- Train and evaluate Logistic Regression and Random Forest models.  
- Provide a single entry point (`main.py`) to run the full pipeline.

 ## 2. Repository Structure

 
```

Predicting-Customer-Churn-in-Telecom/
├── data/
│ ├── raw/ # Original Telco-Customer-Churn.csv
│ └── processed/ # Cleaned / feature-engineered data
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_model_evaluation.ipynb
├── src/
│ ├── init.py
│ ├── data_preprocessing.py # create_data_dirs, load_data, clean_data
│ ├── feature_engineering.py # engineer_features
│ ├── model_training.py # prepare_data, train_logistic_regression, train_random_forest
│ └── evaluation.py # evaluate_model (metrics, confusion matrix)
├── main.py # CLI entry point: full pipeline
├── requirements.txt
└── README.md 

```


## 3. Data Pipeline

The full pipeline (also implemented in `main.py`) is:

1. **Data loading & directories**  
   - `create_data_dirs()` ensures `data/raw` and `data/processed` exist.  
   - `load_data("data/raw/Telco-Customer-Churn.csv")` loads the IBM Telco churn CSV into a pandas DataFrame.[web:71]

2. **Cleaning (`src.data_preprocessing`)**  
   - Strip whitespace from column names and string values.  
   - Convert `TotalCharges` to numeric and handle missing values (e.g., median imputation).  
   - Drop non‑predictive identifiers such as `customerID`.

3. **Feature engineering (`src.feature_engineering`)**  
   - Create `tenure_group` bins to capture customer lifetime patterns.  
   - Create additional risk/flag features (e.g., `HighSpender`, `PaymentRisk`, `HighChurnRisk`) used later in modeling.  
   - Prepare a modeling‑ready DataFrame with interpretable features.

4. **Encoding, split, and scaling (`prepare_data`)**  
   - Map target `Churn` from `"Yes"/"No"` to `1/0`.  
   - Drop `customerID` and `tenure_group` before modeling.  
   - Label‑encode remaining categorical columns.  
   - Split into train/test with stratification on churn, then apply `StandardScaler` to features.

5. **Model training (`src.model_training`)**  
   - `train_logistic_regression(X_train, y_train)` → fitted `LogisticRegression`  
   - `train_random_forest(X_train, y_train)` → fitted `RandomForestClassifier`

6. **Evaluation (`src.evaluation.evaluate_model`)**  
   - For a given model and `(X_test, y_test)`, compute:  
     - Accuracy, precision, recall, F1‑score.  
     - Confusion matrix.  
     - Text classification report.
   - In `04_model_evaluation.ipynb`, additional plots include:
     - Confusion matrix heatmaps.  
     - ROC curves (AUC) for both models.  
     - Feature importance barplot for Random Forest.



## 4. Exploratory Data Analysis (EDA) – Highlights

Detailed EDA is in `01_data_exploration.ipynb` and `02_feature_engineering.ipynb`, but key findings include:[web:48][web:64][web:71]

- **Churn rate**: around one‑quarter to one‑third of customers churn in the IBM Telco dataset.  
- **High‑risk profiles**:
  - Short tenure, high monthly charges.  
  - Month‑to‑month contracts.  
  - Fiber‑optic internet.  
  - Electronic check as payment method.  
- **Feature suitability**:
  - Most engineered and encoded features show useful variation and limited multicollinearity, making them appropriate for modeling.

These insights directly informed the feature engineering and model choice.

---

## 5. How to Run

### 5.1. Setup

```

git clone https://github.com/tundeadekunles1/Predicting-Customer-Churn-in-Telecom.git
cd Predicting-Customer-Churn-in-Telecom

python -m venv .venv

Windows
..venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Place the IBM Telco Customer Churn CSV as:
data/raw/Telco-Customer-Churn.csv

```

### 5.2. Run full pipeline from the command line

python main.py

This will:

- Create data folders if needed.  
- Load, clean, and engineer the dataset.  
- Prepare train/test sets.  
- Train Logistic Regression and Random Forest.  
- Print metrics and classification reports for both models.

### 5.3. Run step‑by‑step in notebooks

Open the `notebooks/` folder and run in order:

1. `01_data_exploration.ipynb` – raw data inspection and basic EDA.  
2. `02_feature_engineering.ipynb` – cleaning + feature engineering → saves processed CSV.  
3. `03_model_training.ipynb` – training Logistic Regression and Random Forest.  
4. `04_model_evaluation.ipynb` – evaluation metrics, confusion matrices, ROC curves, feature importance.

Each notebook is self‑contained and re‑creates any required variables from disk.

---

## 6. Results (High‑Level)

Model‑level numbers will depend slightly on random seeds and implementation details, but typically on this dataset: 

- Logistic Regression achieves solid baseline performance and interpretable coefficients.  
- Random Forest often improves recall and AUC, and its feature importances highlight key churn drivers (contract type, monthly charges, payment method, internet service).
  

These models can be used by marketing and retention teams to prioritize at‑risk customers and design targeted interventions.




