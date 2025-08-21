
# === Paste this at the end of your FINAL BAN620 GROUP PROJECT.ipynb and run it ===
# It saves your best models to the file names the FastAPI app expects.

import joblib, os

# Replace these variables with the actual fitted pipeline objects in your notebook:
#   - boosted_clf_best: your final Boosted Tree CLASSIFIER pipeline (preprocess + model)
#   - regression_tree_best: your final Regression Tree REGRESSOR pipeline (preprocess + model)

# Example names you might already have used (uncomment and adjust as needed):
# boosted_clf_best = boostClassifier_full   # e.g., your GradientBoostingClassifier pipeline
# regression_tree_best = fullRegTree        # e.g., your DecisionTreeRegressor pipeline

os.makedirs("models", exist_ok=True)
joblib.dump(boosted_clf_best, "models/success_clf_latest.pkl")
joblib.dump(regression_tree_best, "models/revenue_reg_latest.pkl")

print("Saved: models/success_clf_latest.pkl (classifier)")
print("Saved: models/revenue_reg_latest.pkl (regressor)")
