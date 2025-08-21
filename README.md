# Predicting Startup Success and Revenue Using Machine Learning

## Project Overview

Startups are inherently risky — while some scale into unicorns, many fail due to funding issues, market misfit, or poor financial management. This project applies **data mining and machine learning techniques** to predict startup **success/failure**, estimate **revenue**, segment startups into clusters, and simulate **investor-startup recommendations**.

Using a dataset of **5,000 global startups** from Kaggle, we applied both **predictive** and **exploratory** approaches to uncover patterns that drive success and sustainable growth.

---

## Project Structure

```
Startup-Success-Prediction
 ┣ data/                  # Dataset files (CSV)
 ┣ notebooks/             # Jupyter notebooks for EDA, modeling, and evaluation
 ┣ models/                # Trained models (pickle/joblib)
 ┣ reports/               # Visualizations and analysis reports
 ┣ requirements.txt       # Required Python libraries
 ┣ README.md              # Project documentation
 ┗ FINAL_REPORT.pdf       # Final project report
```

---

## Tools & Technologies

* **Programming Language:** Python
* **Libraries & Packages:**

  * Data Processing: pandas, numpy
  * Visualization: matplotlib, seaborn
  * Modeling: scikit-learn, statsmodels, keras (for Neural Networks)
  * Clustering & Recommendation: scikit-learn, surprise (collaborative filtering)
* **Environment:** Jupyter Notebook

---

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/startup-success-prediction.git
   cd startup-success-prediction
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

---

## Machine Learning Workflow

1. **Data Collection & Cleaning**

   * Dataset from Kaggle containing startup financial and operational metrics.
   * Handled missing values, normalized numerical features, and encoded categorical variables.

2. **Exploratory Data Analysis (EDA)**

   * Visualized distributions of funding, revenue, burn rate, and retention.
   * Correlation heatmaps to identify key predictors of success and revenue.

3. **Feature Engineering**

   * Derived features like burn efficiency ratio (revenue/burn rate).
   * Created interaction terms (funding × retention, employees × marketing spend).

4. **Model Development**

   * **Classification Models:** Logistic Regression, Decision Trees, Random Forests, kNN, Neural Networks.
   * **Regression Models:** Linear Regression, Random Forest Regressor, Neural Networks.
   * **Clustering:** K-Means for unsupervised segmentation.
   * **Association Rule Mining:** Extracted patterns linking burn rate, funding, and retention.
   * **Recommendation:** Collaborative filtering for investor-startup alignment.

5. **Model Evaluation**

   * Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
   * Regression: R², Mean Absolute Error (MAE), Root Mean Square Error (RMSE).
   * Clustering: Silhouette Score for cluster quality.

6. **Interpretability & Insights**

   * Feature importance analysis from tree-based models.
   * Coefficient analysis in logistic regression for business interpretability.
   * Rule-based insights from association mining.

---

## Model Architectures

### Logistic Regression

* Binary classification (Success vs Failure).
* Provides interpretable coefficients for feature importance.

### Decision Trees & Random Forests

* Tree-based rules for classification and regression.
* Ensemble approach improves accuracy and generalization.

### k-Nearest Neighbors (kNN)

* Predicts based on similarity with closest startups.

### Neural Networks (MLP)

* **Architecture:**

  * Input Layer: Startup features (funding, burn rate, employees, etc.)
  * Hidden Layers: 2–3 dense layers with ReLU activation
  * Output Layer:

    * Sigmoid for classification
    * Linear for regression (revenue prediction)
* Optimizer: Adam
* Regularization: Dropout layers

### K-Means Clustering

* Groups startups into clusters such as high-growth, stable, and at-risk.

### Collaborative Filtering

* Matches startups with investors using feature similarity.

---

## Key Results & Insights

* **Classification (Success/Failure):**

  * Random Forest Accuracy: \~85%
  * Neural Network Accuracy: \~88%

* **Regression (Revenue Prediction):**

  * Random Forest R²: \~0.79
  * Neural Network R²: \~0.82

* **Clustering:** Revealed three primary groups: high-growth, stable performers, and at-risk startups.

* **Association Rules:** High burn rate + low retention = strong predictor of failure.

* **Recommendation System:** Demonstrated potential for data-driven investor-startup alignment.

---

## Dataset

* **Source:** [Startup Failure Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/sakharebharat/startup-failure-prediction-dataset)
* **Features:** funding amount, revenue, burn rate, marketing expenses, employee count, customer retention, and more.

---

## Future Work

* Expand dataset with competitor and market data.
* Deploy models as an interactive web application (Flask/Django).
* Integrate real-time data sources (e.g., Crunchbase, PitchBook).
* Build dashboards in Power BI/Tableau for investor decision support.

---

This project demonstrates how **machine learning can de-risk startup investments** and provide **data-driven insights** for founders, venture capitalists, and analysts.





