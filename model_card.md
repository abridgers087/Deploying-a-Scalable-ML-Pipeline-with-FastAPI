# Model Card — Census Income Classification Model

## **Model Overview**

This model predicts whether an individual earns **">50K" or "<=50K"** based on demographic and employment features from the UCI Census Income dataset. It is trained as part of a machine learning pipeline deployment exercise demonstrating model development, evaluation, versioning, testing, and API deployment using FastAPI and CI/CD.

- **Model Type:** Supervised Classification
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Prediction Output:** Binary income classification
- **Primary Goal:** Demonstrate full ML DevOps integration and model deployment
- **Training details:** RandomForestClassifier(n_estimators=100, random_state=42) trained using an 80/20 split without hyperparameter tuning.

---

## **Intended Use**

### **Primary Uses**

- Educational demonstration of MLOps pipeline
- Example of deploying a trained model via FastAPI
- Show performance evaluation including slice-based fairness testing

### **Users**

- Data science students and engineers
- Instructors evaluating ML pipeline implementation
- Developers exploring CI/CD and inference APIs

### **Out-of-Scope Uses**

- Real hiring or credit decisions
- Financial or legal advice
- Production deployment without monitoring & fairness evaluation

---

## **Training Data**

- **Dataset:** UCI Census Income (`census.csv`)
- **Train/Test Split:** 80/20
- **Features Used:**
  `workclass`, `education`, `marital-status`, `occupation`, `relationship`,  
  `race`, `sex`, `native-country`, plus numeric predictors such as `age`, `hours-per-week`

### **Target Variable**

`salary` — `>50K` or `<=50K`

---

## **Evaluation Metrics**

| Metric    | Value      |
| --------- | ---------- |
| Precision | **0.7419** |
| Recall    | **0.6384** |
| F1 Score  | **0.6863** |

_(Values reflect latest trained model run.)_

---

## **Performance on Data Slices**

Slice metrics evaluate fairness and performance differences for categories such as education, marital-status, workclass, race, and sex.

Results are saved to **`slice_output.txt`**

Example excerpt:

- education: Bachelors, Count: 535
- Precision: 0.7654 | Recall: 0.6101 | F1: 0.6778<br>

Observations:

- Meaningful variation exists across demographic subgroups
- Indicates real-world bias present in training data

---

## **Ethical Considerations**

- Income models risk amplifying existing socioeconomic and demographic inequities
- Dataset reflects historical bias, limiting fairness and reliability
- Should **not** be used for real-world decisions affecting people<br>

---

## **Limitations**

- Limited feature engineering and hyperparameter tuning
- Dataset is outdated and not representative
- CI/CD demonstration pipeline rather than scalable production system
- Not robust to distribution shift<br>

---

## **Recommendations**

- Evaluate fairness metrics continuously when model is deployed
- Use modern representative datasets before real applications
- Add hyperparameter tuning and feature scaling for performance
- Consider additional monitoring for slice-based performance drift<br>

---

## **Contact**

Maintainer: **Ali Bridgers**  
Repository: https://github.com/abridgers087/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

---
