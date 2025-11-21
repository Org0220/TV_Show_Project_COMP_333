"""
TV Show Data Integration Pipeline - Hybrid Chunked Loading
Efficient chunked loading for NDJSON + standard loading for CSVs
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import re
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.db_config import SQLALCHEMY_URL, DATA_PATHS, OUTPUT_PATHS, TABLE_NAMES

# Resolve paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATHS['dataset1'] = str(PROJECT_ROOT / DATA_PATHS['dataset1'])
DATA_PATHS['netflix'] = str(PROJECT_ROOT / DATA_PATHS['netflix'])
DATA_PATHS['disney'] = str(PROJECT_ROOT / DATA_PATHS['disney'])
DATA_PATHS['amazon'] = str(PROJECT_ROOT / DATA_PATHS['amazon'])
DATA_PATHS['hulu'] = str(PROJECT_ROOT / DATA_PATHS['hulu'])
OUTPUT_PATHS['integrated_csv'] = str(PROJECT_ROOT / OUTPUT_PATHS['integrated_csv'])
OUTPUT_PATHS['integration_report'] = str(PROJECT_ROOT / OUTPUT_PATHS['integration_report'])

# --- Ensure output directory exists ---
processed_dir = Path(OUTPUT_PATHS['integrated_csv']).parent
processed_dir.mkdir(parents=True, exist_ok=True)



class TVShowIntegrationHybrid:
    """Hybrid chunked loading integration pipeline"""
    
    CHUNK_SIZE = 2000  # Process 2000 records at a time
    
    def __init__(self):
        self.engine = None
        self.datasets = {}
        self.integration_stats = {}
        logger.info("Initializing Hybrid TV Show Integration Pipeline")
        
    def connect_to_db(self):
        """Connect to PostgreSQL"""
        try:
            self.engine = create_engine(SQLALCHEMY_URL)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL successfully")
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {str(e)[:80]}")
            return False
    
    def load_dataset1_chunked(self):
        """Load Dataset 1 NDJSON in chunks"""
        logger.info("Loading Dataset 1 (NDJSON - chunked)...")
        ndjson_path = DATA_PATHS['dataset1']
        
        try:
            # First pass: count records
            total_records = 0
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                total_records = sum(1 for _ in f)
            logger.info(f"  Total records in file: {total_records:,}")
            
            # Load in chunks
            all_chunks = []
            chunk_num = 0
            current_chunk = []
            
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if idx % 5000 == 0 and idx > 0:
                        logger.info(f"  Loaded {idx:,}/{total_records:,} records...")
                    
                    try:
                        record = json.loads(line)
                        current_chunk.append(record)
                        
                        # Create chunk when it reaches size limit
                        if len(current_chunk) >= self.CHUNK_SIZE:
                            df_chunk = pd.DataFrame(current_chunk)
                            all_chunks.append(df_chunk)
                            chunk_num += 1
                            current_chunk = []
                    except json.JSONDecodeError:
                        continue
            
            # Add remaining records
            if current_chunk:
                df_chunk = pd.DataFrame(current_chunk)
                all_chunks.append(df_chunk)
            
            # Combine all chunks
            if all_chunks:
                df1 = pd.concat(all_chunks, ignore_index=True)
                logger.info(f"Loaded Dataset 1: {df1.shape[0]} rows, {df1.shape[1]} columns")
                self.datasets['dataset1'] = df1
                return df1
            else:
                logger.error("No data loaded from NDJSON")
                return None
                
        except Exception as e:
            logger.error(f"Error loading Dataset 1: {e}")
            return None
    
    def load_csv_datasets(self):
        """Load CSV datasets normally"""
        platforms = ['netflix', 'disney', 'amazon', 'hulu']
        
        for platform in platforms:
            try:
                logger.info(f"Loading {platform.capitalize()}...")
                csv_path = DATA_PATHS[platform]
                df = pd.read_csv(csv_path)
                logger.info(f"  Loaded: {df.shape[0]} rows")
                self.datasets[platform] = df
            except Exception as e:
                logger.error(f"Error loading {platform}: {e}")
    
    def normalize_title(self, title):
        """Normalize title for matching"""
        if pd.isna(title):
            return ""
        title = str(title).strip().lower()
        # Remove country codes
        title = re.sub(r'\s*\(u\.s\.\)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*\(uk\)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*\(us\)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def prepare_datasets(self):
        """Prepare and standardize datasets"""
        logger.info("\nPreparing datasets...")
        
        # Prepare Dataset 1
        if 'dataset1' in self.datasets:
            df1 = self.datasets['dataset1']
            # Use 'name' column as title
            if 'name' in df1.columns:
                df1['title'] = df1['name']
            # Extract year from premiered date
            if 'premiered' in df1.columns:
                df1['release_year'] = pd.to_datetime(df1['premiered'], errors='coerce').dt.year
            df1['title_normalized'] = df1['title'].apply(self.normalize_title)
            df1['release_year'] = df1['release_year'].astype('Int64')  # Nullable integer
            self.datasets['dataset1'] = df1
        
        # Prepare platform datasets
        for platform in ['netflix', 'disney', 'amazon', 'hulu']:
            if platform in self.datasets:
                df = self.datasets[platform]
                
                # Rename genres column
                if 'listed_in' in df.columns:
                    df['genres'] = df['listed_in']
                
                df['title_normalized'] = df['title'].apply(self.normalize_title)
                df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').astype('Int64')
                self.datasets[platform] = df
        
        logger.info("Datasets prepared")
    
    def merge_platform_data(self):
        """Merge platforms with Dataset 1"""
        logger.info("\nMerging platform data...")
        
        main_df = self.datasets['dataset1'].copy()
        
        # Initialize platform flags
        for platform in ['netflix', 'disney', 'amazon', 'hulu']:
            main_df[f'on_{platform}'] = 0
        
        # Match each platform
        for platform in ['netflix', 'disney', 'amazon', 'hulu']:
            if platform not in self.datasets:
                continue
            
            logger.info(f"  Matching {platform.capitalize()}...")
            platform_df = self.datasets[platform]
            
            # Create lookup set for fast matching
            platform_set = set(zip(
                platform_df['title_normalized'],
                platform_df['release_year']
            ))
            
            # Match
            matches = 0
            for idx, row in main_df.iterrows():
                if idx % 2000 == 0 and idx > 0:
                    logger.info(f"    Processed {idx:,} rows...")
                
                key = (row['title_normalized'], row['release_year'])
                if key in platform_set:
                    main_df.at[idx, f'on_{platform}'] = 1
                    matches += 1
            
            self.integration_stats[platform] = matches
            logger.info(f"  {platform.capitalize()}: {matches} matches")
        
        return main_df
    
    def select_output_columns(self, df):
        """Select and rename output columns"""
        logger.info("Selecting output columns...")
        
        # Map Dataset 1 columns to output names
        col_mapping = {
            'name': 'title',
            'genres': 'genres',
            'summary': 'description',
            'premiered': 'premiere_date',
            'release_year': 'release_year',
            'rating': 'rating',
            'language': 'language',
            'type': 'type',
            'runtime': 'runtime_minutes',
            'status': 'status'
        }
        
        # Select available columns
        output_cols = []
        for src_col, dst_col in col_mapping.items():
            if src_col in df.columns:
                output_cols.append((src_col, dst_col))
        
        # Add platform flags
        platform_flags = ['on_netflix', 'on_disney', 'on_amazon', 'on_hulu']
        
        # Create output dataframe
        result_df = pd.DataFrame()
        for src_col, dst_col in output_cols:
            result_df[dst_col] = df[src_col]
        
        for flag in platform_flags:
            if flag in df.columns:
                result_df[flag] = df[flag]

        # ✅ ADD SURROGATE ID COLUMN BEFORE SAVING (UNCHANGED)
        result_df.insert(0, 'id', range(1, len(result_df) + 1))

        logger.info(f"Selected {len(result_df.columns)} columns")
        return result_df
    
    def export_csv(self, df):
        """Export to CSV"""
        try:
            output_path = OUTPUT_PATHS['integrated_csv']
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Exported to CSV: {output_path}")
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            return True
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return False
    
    def save_to_database(self, df):
        """Save to PostgreSQL with proper type handling"""
        try:
            if self.engine is None:
                logger.warning("Database engine not initialized, skipping database save")
                return False
            
            logger.info("Saving to PostgreSQL...")
            
            # Create a copy to avoid modifying the original
            df_db = df.copy()
            
            logger.info(f"  DataFrame shape: {df_db.shape}")
            logger.info(f"  Columns: {list(df_db.columns)}")
            
            # Convert complex objects to JSON strings before saving
            logger.info("  Converting complex objects to JSON strings...")
            if 'genres' in df_db.columns:
                df_db['genres'] = df_db['genres'].apply(lambda x: json.dumps(x) if isinstance(x, list) else (x if isinstance(x, str) else '[]'))
                logger.info("    genres: converted to JSON")
            
            if 'rating' in df_db.columns:
                df_db['rating'] = df_db['rating'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else (x if isinstance(x, str) else '{}'))
                logger.info("    rating: converted to JSON")
            
            # Convert data types to match SQL schema
            type_mapping = {
                'title': 'object',
                'genres': 'object',
                'description': 'object',
                'premiere_date': 'object',
                'release_year': 'Int64',  # Nullable integer
                'rating': 'object',
                'language': 'object',
                'type': 'object',
                'runtime_minutes': 'float64',
                'status': 'object',
                'on_netflix': 'int64',
                'on_disney': 'int64',
                'on_amazon': 'int64',
                'on_hulu': 'int64',
            }
            
            # Apply type conversions
            logger.info("  Applying type conversions...")
            for col, dtype in type_mapping.items():
                if col in df_db.columns:
                    if dtype == 'Int64':
                        df_db[col] = pd.to_numeric(df_db[col], errors='coerce').astype('Int64')
                        logger.info(f"    {col}: converted to Int64")
                    elif dtype == 'float64':
                        df_db[col] = pd.to_numeric(df_db[col], errors='coerce')
                        logger.info(f"    {col}: converted to float64")
                    elif dtype == 'int64':
                        df_db[col] = pd.to_numeric(df_db[col], errors='coerce').astype('int64')
                        logger.info(f"    {col}: converted to int64")
                    elif dtype == 'object':
                        df_db[col] = df_db[col].astype('object')
            
            logger.info(f"  Final data types:")
            for col in df_db.columns:
                logger.info(f"    {col}: {df_db[col].dtype}")
            
            # Insert (recreate) table
            logger.info("  Inserting data into PostgreSQL...")
            df_db.to_sql(
                TABLE_NAMES['integrated'],
                self.engine,
                if_exists='replace',
                index=False,
                chunksize=500,
                method='multi'
            )
            
            # 🚨 IMPORTANT CHANGE: USE engine.begin() SO DDL IS COMMITTED
            # 🚨 ALSO: ENSURE id IS NOT NULL, DROP ANY EXISTING PK, THEN ADD PK
            table = TABLE_NAMES['integrated']
            logger.info("  Enforcing PRIMARY KEY on 'id' with committed DDL...")
            # ✅ BEGIN A TRANSACTION THAT WILL COMMIT
            with self.engine.begin() as conn:  # <<<<<< NEW: BEGIN() COMMITS ON EXIT
                # ✅ ENSURE id NOT NULL (REQUIRED FOR PK)
                conn.execute(text(f'ALTER TABLE {table} ALTER COLUMN id SET NOT NULL;'))
                # ✅ DROP OLD PK IF PRESENT
                conn.execute(text(f'DROP INDEX IF EXISTS {table}_pkey;'))  # harmless if not an index
                conn.execute(text(f'ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_pkey;'))
                # ✅ ADD PRIMARY KEY
                conn.execute(text(f'ALTER TABLE {table} ADD PRIMARY KEY (id);'))
                # ✅ OPTIONAL: MAKE id AUTO-INCREMENT FOR FUTURE INSERTS (SAFE IF ALREADY SET)
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = '{table}'
                          AND column_name = 'id'
                          AND identity_generation IS NOT NULL
                      ) THEN
                        ALTER TABLE {table} ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY;
                      END IF;
                    END$$;
                """))
            logger.info("  ✅ PRIMARY KEY enforced & committed")

            # 🔍 VERIFY: LOG PK STATUS + DUPLICATE CHECK
            with self.engine.connect() as conn:
                # PK name
                pkname = conn.execute(text(f"""
                    SELECT conname
                    FROM pg_constraint
                    WHERE conrelid = 'public.{table}'::regclass
                      AND contype = 'p'
                    LIMIT 1;
                """)).fetchone()
                logger.info(f"  Primary key constraint: {pkname[0] if pkname else 'NONE'}")
                # Duplicate check
                dup = conn.execute(text(f"""
                    SELECT COUNT(*) - COUNT(DISTINCT id) AS duplicate_count
                    FROM {table};
                """)).fetchone()[0]
                logger.info(f"  Duplicate id count: {dup}")

            # Verify row count
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAMES['integrated']}"))
                count = result.fetchone()[0]
                logger.info(f"  Verified: {count:,} rows in PostgreSQL")
                conn.execute(text(f"SELECT 1 FROM {TABLE_NAMES['integrated']} LIMIT 1"))
                logger.info(f"  First row verified successfully")
            
            logger.info(f"Successfully saved {len(df):,} rows to PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_report(self, df):
        """Generate integration report"""
        report_path = OUTPUT_PATHS['integration_report']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TV SHOW DATA INTEGRATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("INTEGRATION OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total shows integrated: {len(df):,}\n")
            f.write(f"Total columns: {len(df.columns)}\n")
            f.write(f"Integration method: Exact title + release year matching\n\n")
            
            # Source datasets
            f.write("SOURCE DATASETS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset 1 (Main): {self.datasets['dataset1'].shape[0]:,} rows\n")
            for platform in ['netflix', 'disney', 'amazon', 'hulu']:
                if platform in self.datasets:
                    f.write(f"{platform.capitalize()}: {self.datasets[platform].shape[0]:,} rows\n")
            f.write("\n")
            
            # Matching results
            f.write("MATCHING RESULTS\n")
            f.write("-" * 80 + "\n")
            for platform, count in self.integration_stats.items():
                if platform in self.datasets:
                    total = self.datasets[platform].shape[0]
                    pct = (count / total * 100) if total > 0 else 0
                    f.write(f"{platform.capitalize()}: {count:,} / {total:,} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Platform distribution
            f.write("PLATFORM AVAILABILITY\n")
            f.write("-" * 80 + "\n")
            for platform in ['netflix', 'disney', 'amazon', 'hulu']:
                col = f'on_{platform}'
                if col in df.columns:
                    count = (df[col] == 1).sum()
                    pct = (count / len(df) * 100)
                    f.write(f"{platform.capitalize()}: {count:,} shows ({pct:.1f}%)\n")
            f.write("\n")
            
            # Column info
            f.write("OUTPUT COLUMNS\n")
            f.write("-" * 80 + "\n")
            for col in df.columns:
                non_null = df[col].notna().sum()
                null_pct = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
                f.write(f"{col}: {non_null:,} non-null ({100-null_pct:.1f}% coverage)\n")
            f.write("\n")
            
            # Notes
            f.write("NOTES\n")
            f.write("-" * 80 + "\n")
            f.write("- Matching: Exact title (normalized) + release year\n")
            f.write("- Normalization: lowercase, remove country codes, normalize whitespace\n")
            f.write("- Platform flags: 1 = available, 0 = not available or unmatched\n")
            f.write("- Data integration preserves all original data from Dataset 1\n")
        
        logger.info(f"Report generated: {report_path}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING HYBRID TV SHOW INTEGRATION PIPELINE")
        logger.info("="*80 + "\n")
        
        try:
            # Connect to DB
            self.connect_to_db()
            
            # Load datasets
            logger.info("="*80)
            logger.info("LOADING DATASETS")
            logger.info("="*80 + "\n")
            self.load_dataset1_chunked()
            self.load_csv_datasets()
            
            # Validate
            if 'dataset1' not in self.datasets or self.datasets['dataset1'] is None:
                logger.error("ERROR: Failed to load Dataset 1. Stopping.")
                return
            
            logger.info(f"\nDatasets loaded: {len(self.datasets)} total")
            
            # Prepare
            self.prepare_datasets()
            
            # Merge
            logger.info("\n" + "="*80)
            logger.info("MERGING DATASETS")
            logger.info("="*80 + "\n")
            integrated_df = self.merge_platform_data()
            
            # Process output
            logger.info("\n" + "="*80)
            logger.info("PROCESSING OUTPUT")
            logger.info("="*80 + "\n")
            output_df = self.select_output_columns(integrated_df)
            
            # Export
            self.export_csv(output_df)
            
            # Save to DB
            logger.info("\n" + "="*80)
            logger.info("DATABASE SAVE")
            logger.info("="*80 + "\n")
            
            if self.engine:
                success = self.save_to_database(output_df)
                if success:
                    logger.info("✅ Database save successful\n")
                else:
                    logger.warning("⚠️  Database save failed, but CSV export succeeded\n")
            else:
                logger.warning("⚠️  Database connection not available\n")
            
            # Report
            self.generate_report(output_df)
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80 + "\n")
            logger.info(f"Output files:")
            logger.info(f"  - CSV: {OUTPUT_PATHS['integrated_csv']}")
            logger.info(f"  - Report: {OUTPUT_PATHS['integration_report']}")
            logger.info(f"  - Database: {TABLE_NAMES['integrated']} (PostgreSQL)")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == '__main__':
    integration = TVShowIntegrationHybrid()
    integration.run_pipeline()
