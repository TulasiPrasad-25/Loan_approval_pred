# 🏦 Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be approved or rejected based on applicant financial and demographic data. Built end-to-end — from raw data preprocessing to model selection and feature engineering.

---

## 📌 Problem Statement

Financial institutions process thousands of loan applications and need a reliable, data-driven way to assess risk. This project builds a binary classification model that predicts loan approval outcomes, with **Precision** as the primary evaluation metric — minimizing false approvals is critical in the lending domain.

---

## 📂 Dataset

| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of the primary applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Credit_Score` | Applicant's credit score |
| `DTI_Ratio` | Debt-to-income ratio |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Duration of the loan (months) |
| `Savings` | Applicant's savings |
| `Collateral_Value` | Value of collateral provided |
| `Employment_Status` | Salaried / Self-employed |
| `Marital_Status` | Single / Married |
| `Education_Level` | Graduate / Not Graduate |
| `Gender` | Male / Female |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Purpose` | Personal / Car / Business, etc. |
| `Loan_Approved` | **Target** — Yes / No |

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

---

## 🚀 Project Workflow

### 1. Data Preprocessing
- Identified and handled missing values using `SimpleImputer`
  - **Numerical columns** → filled with **mean**
  - **Categorical columns** → filled with **most frequent** value
- Removed the `Applicant_ID` column (non-informative identifier)
- Verified no duplicate records

### 2. Exploratory Data Analysis (EDA)
- **Class distribution** → Pie chart of approved vs. rejected loans
- **Gender distribution** → Bar plot
- **Income analysis** → Histograms for applicant & co-applicant income
- **Outlier detection** → Box plots for Income, Credit Score, DTI Ratio, Savings vs Loan Approval
- **Correlation Heatmap** → Identified top predictors of loan approval

### 3. Encoding
- **Label Encoding** → Binary columns (`Education_Level`, `Loan_Approved`)
- **One-Hot Encoding** → Multi-class columns (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`)

### 4. Train-Test Split & Feature Scaling
- 80/20 train-test split with `random_state=42`
- Applied `StandardScaler` to normalize features

### 5. Model Training & Evaluation
Three models were trained and compared:

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | - | - | - | - |
| K-Nearest Neighbors | - | - | - | - |
| **Naïve Bayes** ✅ | **Best** | - | - | - |

> ✅ **Naïve Bayes** was selected as the final model based on highest **Precision** score.

### 6. Feature Engineering
- Added polynomial features: `DTI_Ratio²` and `Credit_Score²`
- Re-trained Naïve Bayes on engineered features for improved performance

---

## 📊 Key Insights from EDA

- **Credit Score** is among the strongest predictors of loan approval
- Applicants with lower **DTI Ratio** are significantly more likely to get approved
- **Urban** applicants show higher approval rates compared to Rural
- Income alone is not a strong predictor — savings and collateral matter more

---

## 🗂️ Project Structure

```
loan-approval-prediction/
│
├── loan_approval_data.csv       # Dataset
├── loan_project.ipynb           # Main notebook (EDA + Modeling)
└── README.md                    # Project documentation
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Launch the notebook
jupyter notebook loan_project.ipynb
```

---

## 🙋 Author

**Tulasi Prasad**
B.Tech — Computer Science & AI | Siddharth Institute of Engineering and Technology
📧 tulasiprasad2526@gmail.com

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
