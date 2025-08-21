# Step 9: Deploy the Best Model (Super Simple)

## Do these 3 things
1) **Save your best pipelines** from your notebook:
   ```python
   import joblib
   joblib.dump(best_success_pipeline, "models/success_clf_latest.pkl")
   joblib.dump(best_revenue_pipeline, "models/revenue_reg_latest.pkl")
   ```
2) **(Optional) Edit the input fields** in `app.py -> class Features` so they match your training columns (names & order).
3) **Run the service**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```

### Endpoints
- `GET /health`
- `POST /predict/success`
- `POST /predict/revenue`

Example:
```bash
curl -X POST http://localhost:8080/predict/success   -H "Content-Type: application/json"   -d '{"Funding_Amount": 2000000, "Employees_Count": 18, "Burn_Rate": 0.12, "Customer_Retention_Rate": 0.84, "Marketing_Expense": 120000}'
```

## Docker
```bash
docker build -t startup-ml:latest .
docker run -p 8080:8080 startup-ml:latest
```

## Notes
- Put your real `.pkl` files in `./models/` as named above (or update `models/registry.json`).
- Keep the **feature order** identical to training.
- If you only have one task, you can remove the other endpoint.
- Add any extra features you used to `Features` and the NumPy array in the same order.
