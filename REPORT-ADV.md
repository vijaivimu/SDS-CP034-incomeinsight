# 📄 IncomeInsight – Project Report - 🔴 **Advanced Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Phase 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What features show the strongest correlation with earning >$50K?

### 🔑 Question 2: How does income vary with education, marital status, or hours worked per week?

### 🔑 Question 3: Are there disparities across race, sex, or native country?

### 🔑 Question 4: Do capital gains/losses strongly impact the income label?

---

## ✅ Phase 2: Model Development

> This phase spans 3 weeks. Answer each set of questions weekly as you build, train, evaluate, and improve your models.

---

### 📆 Week 1: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
**Which high-cardinality categorical features (e.g., `occupation`, `native_country`) are best suited for embeddings, and how did you determine the embedding dimensions for each?**

💡 **Hint:**  
Use `.nunique()` to assess cardinality.  
Use heuristics like `min(50, (n_unique + 1) // 2)` for embedding dimension.  
Consider category frequency: are there rare classes that may cause overfitting?

✏️ *Your answer here...*

---

#### 🔑 Question 2:
**What preprocessing steps did you apply to the numerical features before feeding them into your FFNN, and why are those steps important for deep learning models?**

💡 **Hint:**  
Inspect `df.describe()` and histograms.  
Apply `StandardScaler`, `MinMaxScaler`, or log transformations based on spread and skew.  
Avoid scaling label-encoded categorical values.

✏️ *Your answer here...*

---

#### 🔑 Question 3:
**Did you create any new features or interactions, and what evidence suggests they might improve predictive performance?**

💡 **Hint:**  
Visualize combinations of features across income brackets.  
Use correlation with the target, separation by class, or logic from social/economic context.  
Try binary flags or ratios.

✏️ *Your answer here...*

---

#### 🔑 Question 4:
**Which features (if any) did you decide to exclude from the model input, and what was your reasoning?**

💡 **Hint:**  
Drop features with very low variance, high missingness, or high correlation to others.  
Ask: Does this feature introduce noise or offer little predictive power?

✏️ *Your answer here...*

---

#### 🔑 Question 5:
**What is the distribution of the target class in your dataset, and how might this class imbalance affect your model’s learning and evaluation?**

💡 **Hint:**  
Use `.value_counts(normalize=True)` to check balance of >50K vs ≤50K.  
Class imbalance may require:
- Stratified sampling  
- Weighted loss functions  
- Evaluation via precision, recall, F1, and AUC instead of just accuracy.

✏️ *Your answer here...*


---

### 📆 Week 2: Model Development & Experimentation

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

### 📆 Week 3: Model Tuning

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✅ Phase 3: Model Deployment

> Document your approach to building and deploying the Streamlit app, including design decisions, deployment steps, and challenges.

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---
