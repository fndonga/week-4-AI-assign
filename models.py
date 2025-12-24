# ============================================================
# AI APPLICATIONS IN SOFTWARE ENGINEERING
# CONSOLIDATED PRACTICAL IMPLEMENTATION
# ============================================================
# This script demonstrates:
# 1. AI-powered code completion comparison
# 2. Automated testing logic (Selenium-style simulation)
# 3. Predictive analytics using Machine Learning
# ============================================================


# =========================
# SECTION 1: IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd

# Machine Learning libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# (Optional) Visualization
import matplotlib.pyplot as plt


# ============================================================
# TASK 1: AI-POWERED CODE COMPLETION
# ============================================================
# Objective:
# Compare manual code vs AI-generated code for sorting a list
# of dictionaries by a specific key.
# ============================================================

print("\n========== TASK 1: AI-POWERED CODE COMPLETION ==========")

# Sample data: list of dictionaries
data = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]

# ---------
# Manual Implementation
# ---------
def sort_dicts_manual(data, key):
    """
    Manually sorts a list of dictionaries using a specific key.
    This version assumes the key always exists.
    """
    return sorted(data, key=lambda x: x[key])

# ---------
# AI-Generated (Copilot-style) Implementation
# ---------
def sort_dicts_ai(data, key, reverse=False):
    """
    AI-enhanced version:
    - Uses .get() to prevent KeyError
    - Allows reverse sorting
    - More flexible and robust
    """
    return sorted(data, key=lambda item: item.get(key), reverse=reverse)

# Demonstration
print("Manual sort:", sort_dicts_manual(data, "score"))
print("AI sort:", sort_dicts_ai(data, "score", reverse=True))


# ============================================================
# TASK 2: AUTOMATED TESTING WITH AI (SIMULATED)
# ============================================================
# Objective:
# Demonstrate automated testing logic similar to Selenium/Testim
# NOTE: This is a logical simulation for environments without a browser.
# ============================================================

print("\n========== TASK 2: AUTOMATED TESTING (SIMULATION) ==========")

# Simulated login function
def login(username, password):
    """
    Simulates a login system.
    """
    if username == "admin" and password == "password123":
        return "Login Successful"
    else:
        return "Login Failed"

# Automated test cases
test_cases = [
    ("admin", "password123"),   # valid credentials
    ("admin", "wrongpass"),     # invalid password
    ("user", "password123")     # invalid username
]

# Execute tests automatically
for user, pwd in test_cases:
    result = login(user, pwd)
    print(f"Test login with ({user}, {pwd}): {result}")

# Explanation:
# AI-driven testing tools automatically generate, execute,
# and maintain such test cases, increasing coverage and reliability.


# ============================================================
# TASK 3: PREDICTIVE ANALYTICS FOR RESOURCE ALLOCATION
# ============================================================
# Objective:
# Use Machine Learning to simulate predictive decision-making
# using the Breast Cancer Dataset (Kaggle-equivalent).
# ============================================================

print("\n========== TASK 3: PREDICTIVE ANALYTICS ==========")

# Load dataset
dataset = load_breast_cancer()
X = dataset.data      # Features
y = dataset.target    # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model F1 Score: {f1:.2f}")

# Optional visualization: Feature importance
importances = model.feature_importances_
plt.figure(figsize=(10,4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()


# ============================================================
# ETHICAL NOTE (DOCUMENTED IN REPORT)
# ============================================================
# Potential bias may arise if training data underrepresents
# certain groups. Fairness tools such as IBM AI Fairness 360
# can help detect and mitigate such bias.
# ============================================================


# ============================================================
# END OF SCRIPT
# ============================================================
print("\nAll AI software engineering tasks executed successfully.")
