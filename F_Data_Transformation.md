# F_Data_Transformation.md
# Data Transformation Documentation

This document describes the data-transformation phase of the TV Show Project.  
It is the third stage of the ETL workflow, following Data Integration and Data Preparation.

The goal of this stage is to convert the cleaned dataset into a machine-learning–ready form by extracting numerical features, normalizing values, and selecting only relevant columns for modeling tasks such as genre prediction.

---

## 1. Input and Output

**Input File:**
- `data/processed/cleaned_tv_shows.csv`

**Output Files:**
- `data/processed/transformed_tv_shows.csv`
- `data/processed/data_transformation_report.txt`
- PostgreSQL table: `integrated_tv_shows_transformed`

---

## 2. Objectives of Data Transformation

The transformation stage performs three major tasks:

### 2.1 Date Encoding  
Converts the `premiere_date` field into multiple numeric columns used by ML models:

| New Column | Description |
|------------|-------------|
| `premiere_year` | Extracted year |
| `premiere_month` | Extracted month |
| `premiere_dayofweek` | 0=Monday ... 6=Sunday |
| `premiere_is_weekend` | 1 if Saturday/Sunday else 0 |
| `premiere_decade` | Decade grouping (e.g., 1990, 2000, 2010) |

This allows models to learn seasonality or temporal patterns in genre distribution.

---

### 2.2 Numeric Feature Scaling  

Three forms of scaling are applied to the following numeric columns:
- `runtime_minutes`
- `rating_avg`
- `release_year`

The script generates **three scaled versions** of each:

| Scaling Method | Formula | Purpose |
|----------------|---------|---------|
| Min-Max Scaling | (x - min) / (max - min) | Normalizes to [0,1] |
| Z-Score Normalization | (x - mean) / std | Standardized feature distribution |
| Decimal Scaling | x / (10^j) | Moves decimal until max(|x|) < 1 |

The project uses **Z-score scaled columns** for modeling, but retains all three in the CSV for transparency.

---

### 2.3 Feature Selection for Genre Prediction

To reduce noise and avoid multicollinearity, only fields relevant to ML modeling are kept.

Columns that are **kept**:

| Column | Purpose |
|--------|---------|
| `id` | Primary key |
| `title_normalized` | Clean text label for potential text-based models |
| `language` | Categorical feature |
| `type` | Useful for identifying show formats |
| `status` | Whether the show is running or ended |
| `runtime_minutes_zscore` | Numeric feature |
| `rating_avg_zscore` | Numeric feature |
| `release_year_zscore` | Numeric feature |
| `premiere_year` | Encoded date feature |
| `premiere_month` | Encoded date feature |
| `premiere_dayofweek` | Encoded date feature |
| `premiere_is_weekend` | Encoded date feature |
| `premiere_decade` | Encoded date feature |
| `genres_parsed` | Machine-learning label (multi-label classification) |

Columns from the cleaned dataset that are dropped:

| Dropped Column | Reason |
|----------------|--------|
| `title` | Redundant; use `title_normalized` |
| `rating` | Raw nested metadata; replaced by `rating_avg` |
| `genres` | Raw form; replaced by `genres_parsed` |
| `description` | Not used in current models |
| `premiere_date` | Replaced by encoded features |
| `runtime_minutes` | Raw value; replaced by scaled versions |
| `release_year` | Raw value; replaced by scaled version |

---

## 3. Database Output

After feature engineering, the final ML-ready dataset is written to PostgreSQL:

