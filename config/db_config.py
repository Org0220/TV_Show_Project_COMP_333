"""
PostgreSQL Database Configuration
"""

# PostgreSQL Connection Configuration
DATABASE_CONFIG = {
    'host': '127.0.0.1',
    'port': 5433,
    'database': 'tv_show_db',
    'user': 'postgres',
    'password': 'postgres',  # Change this to your PostgreSQL password
}

# SQLAlchemy Connection String
SQLALCHEMY_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

# Data Paths
DATA_PATHS = {
    'dataset1': 'data/raw/the_500mb_tv_show_dataset.ndjson',
    'netflix': 'data/raw/netflix_titles.csv',
    'disney': 'data/raw/disney_plus_titles.csv',
    'amazon': 'data/raw/amazon_prime_titles.csv',
    'hulu': 'data/raw/hulu_titles.csv',
}

# Output Paths
OUTPUT_PATHS = {
    'integrated_csv': 'data/processed/integrated_tv_shows.csv',
    'integration_report': 'data/processed/integration_report.txt',
    'data_dictionary': 'data/processed/data_dictionary.md',
}

# Fuzzy Matching Configuration
FUZZY_MATCH_CONFIG = {
    'threshold': 0.85,
    'method': 'token_set_ratio',  # Options: 'ratio', 'token_sort_ratio', 'token_set_ratio'
}

# Table Names
TABLE_NAMES = {
    'raw_dataset1': 'raw_dataset1',
    'raw_netflix': 'raw_netflix',
    'raw_disney': 'raw_disney',
    'raw_amazon': 'raw_amazon',
    'raw_hulu': 'raw_hulu',
    'integrated': 'integrated_tv_shows',
}
