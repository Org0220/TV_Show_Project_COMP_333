# G - Modeling

## Overview
The **Modeling** phase focuses on building machine learning models to predict the genre of a TV show. Specifically, we perform **Binary Classification** to predict if a show belongs to the **"Comedy"** genre.

We compare the performance of three different models on two versions of the dataset:
1.  **Raw Data**: Minimal preprocessing (imputation and encoding) applied to the integrated dataset.
2.  **Transformed Data**: The output of the Data Transformation phase, which includes scaling, feature selection, and date encoding.

## Target Variable
-   **Target**: `Is Comedy?` (1 = Yes, 0 = No)
-   **Source**: Extracted from the `genres` column.

## Models Used
1.  **Logistic Regression**: A baseline linear model.
2.  **Random Forest Classifier**: An ensemble method using decision trees.
3.  **Gradient Boosting Classifier**: A boosting method for improved accuracy.

## Running the Modeling Script
To run the modeling pipeline, execute the following command:

```bash
python scripts/modeling_final.py
```

## Outputs
-   **Report**: `data/processed/modeling_report.txt` containing Accuracy, Precision, Recall, and F1-Score for all model/data combinations.
-   **Console Output**: Real-time training progress and summary metrics.

## Results Summary
(Example results from initial run)

| Dataset | Model | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- |
| **Raw** | Logistic Regression | ~84% | ~0.62 |
| **Raw** | Random Forest | ~83% | ~0.62 |
| **Transformed** | Logistic Regression | ~83% | ~0.60 |

*Note: Raw data performed comparably to or slightly better than transformed data for this specific task, suggesting that the raw categorical features (like 'status' and 'type') are strong predictors even without complex scaling.*
