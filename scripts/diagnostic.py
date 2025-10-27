#!/usr/bin/env python3
"""
Database Diagnostic Script
Check PostgreSQL connection and table contents
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, inspect

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import SQLALCHEMY_URL, TABLE_NAMES, OUTPUT_PATHS, DATA_PATHS

print("="*80)
print("DATABASE DIAGNOSTIC TOOL")
print("="*80)
print()

# Test 1: Connection
print("TEST 1: Database Connection")
print("-"*80)
try:
    engine = create_engine(SQLALCHEMY_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL Connection: SUCCESS")
except Exception as e:
    print(f"❌ PostgreSQL Connection: FAILED")
    print(f"   Error: {e}")
    sys.exit(1)

print()

# Test 2: List all tables
print("TEST 2: Tables in Database")
print("-"*80)
try:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables in tv_show_db: {len(tables)}")
    for table in tables:
        print(f"  • {table}")
    
    if TABLE_NAMES['integrated'] in tables:
        print(f"✅ integrated_tv_shows table EXISTS")
    else:
        print(f"❌ integrated_tv_shows table NOT FOUND")
except Exception as e:
    print(f"❌ Error listing tables: {e}")

print()

# Test 3: Check integrated_tv_shows contents
print("TEST 3: Table Contents - integrated_tv_shows")
print("-"*80)
try:
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) as row_count FROM {TABLE_NAMES['integrated']}"))
        row_count = result.fetchone()[0]
        print(f"Row count: {row_count}")
        
        if row_count == 0:
            print("❌ Table is EMPTY - No data found!")
        else:
            print(f"✅ Table has {row_count} records")
            
            # Show first few rows
            df = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['integrated']} LIMIT 5", engine)
            print("\nFirst 5 rows:")
            print(df.to_string())
            
            print("\nColumn names:")
            for col in df.columns:
                print(f"  • {col}")
except Exception as e:
    print(f"❌ Error querying table: {e}")
    print(f"   (Table may not exist yet)")

print()

# Test 4: Check CSV file
print("TEST 4: CSV Output File")
print("-"*80)
try:
    csv_path = PROJECT_ROOT / OUTPUT_PATHS['integrated_csv']
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024**2)
        df = pd.read_csv(csv_path, nrows=5)
        print(f"✅ CSV file exists: {csv_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Rows: {len(pd.read_csv(csv_path))}")
        print(f"   Columns: {len(df.columns)}")
        print(f"\n   First row:")
        print(f"   {df.iloc[0].to_dict()}")
    else:
        print(f"❌ CSV file NOT found: {csv_path}")
except Exception as e:
    print(f"❌ Error reading CSV: {e}")

print()

# Test 5: Database vs CSV comparison
print("TEST 5: Database vs CSV Comparison")
print("-"*80)
try:
    # Count in database
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAMES['integrated']}"))
        db_count = result.fetchone()[0]
    
    # Count in CSV
    csv_path = PROJECT_ROOT / OUTPUT_PATHS['integrated_csv']
    csv_count = len(pd.read_csv(csv_path))
    
    print(f"Records in Database: {db_count}")
    print(f"Records in CSV:      {csv_count}")
    
    if db_count == csv_count:
        print("✅ Counts match!")
    elif db_count == 0:
        print("⚠️  Database is empty but CSV has data")
    else:
        print(f"⚠️  Count mismatch: {csv_count - db_count} records difference")
except Exception as e:
    print(f"❌ Error comparing: {e}")

print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
