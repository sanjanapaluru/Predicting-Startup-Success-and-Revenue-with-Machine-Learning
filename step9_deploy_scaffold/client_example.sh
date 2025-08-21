
#!/usr/bin/env bash
set -euo pipefail

echo "Health:"
curl -s http://localhost:8080/health | jq . || curl -s http://localhost:8080/health

echo
echo "Success prediction:"
curl -s -X POST http://localhost:8080/predict/success   -H "Content-Type: application/json"   -d '{"Funding_Amount": 2000000, "Employees_Count": 18, "Burn_Rate": 0.12, "Customer_Retention_Rate": 0.84, "Marketing_Expense": 120000}' | jq . || true

echo
echo "Revenue prediction:"
curl -s -X POST http://localhost:8080/predict/revenue   -H "Content-Type: application/json"   -d '{"Funding_Amount": 2000000, "Employees_Count": 18, "Burn_Rate": 0.12, "Customer_Retention_Rate": 0.84, "Marketing_Expense": 120000}' | jq . || true
