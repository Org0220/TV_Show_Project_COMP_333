# TV_Show_Project_COMP_333

TV show data integration and analysis project combining NDJSON source data with platform availability (Netflix, Disney+, Amazon, Hulu).

## Setup

### Prerequisites
- Python 3.12+
- PostgreSQL 17
- pandas, numpy, sqlalchemy, psycopg2

### Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Update `config/db_config.py` with your PostgreSQL credentials:
   ```python
   SQLALCHEMY_URL = "postgresql://username:password@localhost:5432/tv_show_db"
   ```

## 📁 Datasets to be Integrated

| # | Dataset | Link | Main Focus | Key Columns |
|---|----------|------|-------------|--------------|
| 1 | **The 500MB TV Show Dataset** | [https://www.kaggle.com/datasets/iyadelwy/the-500mb-tv-show-dataset](https://www.kaggle.com/datasets/iyadelwy/the-500mb-tv-show-dataset) | Detailed metadata for global TV shows | `id`,`title`, `description`, `genres`, `language`, `country`, `rating`, `release_year`, `cast`, `production_company` |
| 2 | **Netflix Shows** | [https://www.kaggle.com/datasets/shivamb/netflix-shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) | Titles on Netflix | `show_id`, `title`, `type`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description` |
| 3 | **Disney+ Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows) | Titles on Disney+ | Similar schema as Netflix dataset |
| 4 | **Amazon Prime Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows) | Titles on Amazon Prime | Similar schema as Netflix dataset |
| 5 | **Hulu Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows) | Titles on Hulu | Similar schema as Netflix dataset |

### ⚠️ Download the datasets and place them in `/data/raw`

---

## Running the Integration

### Full Pipeline
```bash
python scripts/integration_final.py
```

Outputs:
- CSV: `data/processed/integrated_tv_shows.csv` (10,200 rows × 14 columns)
- Database: `integrated_tv_shows` table in PostgreSQL
- Report: `data/processed/integration_report.txt`

### Diagnostic Tools
```bash
python scripts/diagnostic.py          # Verify database connection and data integrity
python scripts/analyze_dependencies.py # Analyze functional dependencies
python scripts/reset_db.py            # Reset database for testing
```

## Integration Method

**Exact Matching**: (Title, Release Year)
- Normalizes titles: lowercase, remove country codes, normalize whitespace
- Matches against Netflix, Disney+, Amazon, Hulu datasets
- Platform availability stored as binary flags (1=available, 0=not)
- Adds Primary Key id 

## Key Statistics

- **Total Shows**: 10,200
- **Platform Coverage**: Hulu (5.5%), Netflix (3.4%), Amazon (1.4%), Disney+ (0.9%)
- **Unmatched**: 89.1% of shows not found in any platform dataset
- **Data Quality**: 99%+ coverage for critical fields

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| title | TEXT | Show title |
| genres | JSON | Genre array |
| description | TEXT | Show description |
| premiere_date | DATE | First air date |
| release_year | INTEGER | Release year |
| rating | JSON | Rating metadata |
| language | TEXT | Primary language |
| type | TEXT | Show type (Scripted, Animation, etc.) |
| runtime_minutes | FLOAT | Episode runtime |
| status | TEXT | Status (Ended, Running, etc.) |
| on_netflix, on_disney, on_amazon, on_hulu | INTEGER | Platform availability (0 or 1) |

## Documentation

- `C_Data_Integration.md` - Integration process, examples, and functional dependency analysis
- `C_Data_Preparation.md` - Data preparation steps, null handling, duplicate removal, outlier detection, and examples

## Data Preparation

After integration, a dedicated data-preparation step standardises fields, handles missing values, removes duplicates, and flags outliers. The implementation is provided in `scripts/data_preparation_final.py` and is intended to be run after `integration_final.py`.

What the script does
- Loads: `data/processed/integrated_tv_shows.csv` (integration output)
- Produces:
   - `data/processed/cleaned_tv_shows.csv` (deduplicated, transformed)
   - `data/processed/data_preparation_report.txt` (plain-text report with null counts, examples, duplicate, and outlier stats)
- Key actions:
   - Normalize titles into `title_normalized` (lowercase, remove common country codes, normalize whitespace)
   - Parse and extract numeric `rating_avg` from `rating` metadata
   - Parse `genres` into `genres_parsed` (list)
   - Coerce `release_year` to nullable integer and parse `premiere_date` to datetime
   - Impute `runtime_minutes` nulls with the column median (robust choice)
   - Fill missing `language` with `Unknown`
   - Detect and remove duplicates by composite key (`title_normalized` + `release_year`) keeping the first occurrence
   - Flag outliers (IQR rule) on numeric fields (`runtime_minutes`, `rating_avg`) and report examples; no automatic capping/removal is performed by default

Quick run
```bash
python scripts/data_preparation_final.py
```

Notes / rationale
- The script is intentionally conservative: ratings (`rating_avg`) are extracted but left NaN where missing to avoid bias; outliers are flagged rather than removed to enable domain review.
- The report written to `data/processed/data_preparation_report.txt` contains exact counts and short examples you can paste into your final report.

## Project Structure

```
├── config/
│   └── db_config.py              # Database configuration
├── data/
│   ├── raw/                       # Source datasets
│   └── processed/                 # Output CSV and reports
├── scripts/
│   ├── integration_final.py       # Main integration pipeline
│   ├── diagnostic.py              # Database diagnostics
│   ├── analyze_dependencies.py    # Dependency analysis
│   └── reset_db.py                # Database utilities
└── C_Data_Integration.md          # Integration documentation
```
##  Tools Planned for the Project

This project combines multiple open datasets on TV shows and streaming platforms. To support data collection, integration, transformation, and analysis, we use the following tools and systems:

| Category | Tool | Purpose |
|-----------|----------------|----------|
| **Development Environment** | **VS Code**, **Google Colab** | Used for development, prototyping, and collaborative testing. |
| **Programming Language** | **Python 3.12** | Main language for data ingestion, integration, transformation, and analysis. |
| **Database System** | **PostgreSQL 17** | Stores the integrated TV show data and supports SQL queries and ETL operations through SQLAlchemy. |
| **Data Integration & ETL Libraries** | **pandas**, **numpy**, **sqlalchemy**, **psycopg2** | Handle merging, cleaning, and loading of datasets into the PostgreSQL database. |
| **Data Cleaning & Preparation** | **pandas**, **feature-engine**, **scikit-learn** | Manage missing values, normalization, and transformation of numeric and categorical features. |
| **Data Visualization** | **Matplotlib**, **Seaborn**, **Power BI (optional)** | Used for exploratory data analysis and to create visual summaries of platform coverage and trends. |
| **Modeling / Evaluation (Phase 2)** | **scikit-learn** | Builds and evaluates classification models (e.g., genre or rating prediction) using metrics such as accuracy and F1-score. |
| **Version Control** | **Git + GitHub** | For source control, documentation, and team collaboration. |

**Why choose these tools?:**  
These tools ensure an efficient and reproducible workflow for ETL, data quality management, and machine learning. PostgreSQL provides structured data storage, while Python libraries like pandas and scikit-learn allow for flexible data manipulation and model experimentation. Visualization tools help us better understand platform availability, ratings, and genre distributions.

```
