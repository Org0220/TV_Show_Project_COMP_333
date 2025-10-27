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
| 1 | **The 500MB TV Show Dataset** | [https://www.kaggle.com/datasets/iyadelwy/the-500mb-tv-show-dataset](https://www.kaggle.com/datasets/iyadelwy/the-500mb-tv-show-dataset) | Detailed metadata for global TV shows | `title`, `description`, `genres`, `language`, `country`, `rating`, `release_year`, `cast`, `production_company` |
| 2 | **Netflix Shows** | [https://www.kaggle.com/datasets/shivamb/netflix-shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) | Titles on Netflix | `show_id`, `title`, `type`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description` |
| 3 | **Disney+ Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows) | Titles on Disney+ | Similar schema as Netflix dataset |
| 4 | **Amazon Prime Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows) | Titles on Amazon Prime | Similar schema as Netflix dataset |
| 5 | **Hulu Movies and TV Shows** | [https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows](https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows) | Titles on Hulu | Similar schema as Netflix dataset |

> Download the datasets and place them in `/data/raw`

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

**Exact Matching**: (Title, Release Year) composite key
- Normalizes titles: lowercase, remove country codes, normalize whitespace
- Matches against Netflix, Disney+, Amazon, Hulu datasets
- Platform availability stored as binary flags (1=available, 0=not)

## Key Statistics

- **Total Shows**: 10,200
- **Platform Coverage**: Hulu (5.5%), Netflix (3.4%), Amazon (1.4%), Disney+ (0.9%)
- **Unmatched**: 89.1% of shows not found in any platform dataset
- **Data Quality**: 99%+ coverage for critical fields

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
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
