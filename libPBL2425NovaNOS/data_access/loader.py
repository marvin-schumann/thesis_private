# data_pipeline_nos/data_access/loader.py
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Dict, Any
# from ..config import settings # Relative import if running as part of the package
from config import settings # For direct script execution or testing from root

def get_db_engine():
    """Creates and returns a SQLAlchemy engine."""
    engine_url = f'postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}'
    return create_engine(engine_url)

def fetch_table_from_db(table_name: str, engine) -> pd.DataFrame:
    """Fetches a complete table from the database."""
    query = f"SELECT * FROM {settings.DB_SCHEMA}.{table_name};"
    return pd.read_sql_query(query, engine)

def load_and_concatenate_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Loads all raw tables from the database and concatenates
    client and call tables.
    Returns a dictionary of the three base DataFrames:
    'clients_df', 'calls_df', 'masterdatagcs_df'.
    """
    engine = get_db_engine()
    print("Fetching tables from database...")

    table_names = [
        'masterdataclients_jan_may', 'masterdataclients_jun_oct',
        'mastercalls_jan_feb', 'mastercalls_mar_apr',
        'mastercalls_may_jul', 'mastercalls_aug_oct',
        'masterdatagcs'
    ]

    raw_tables = {}
    for table_name in table_names:
        print(f"Fetching {table_name}...")
        raw_tables[table_name] = fetch_table_from_db(table_name, engine)

    print("Concatenating client data...")
    clients_df = pd.concat(
        [raw_tables['masterdataclients_jan_may'], raw_tables['masterdataclients_jun_oct']],
        ignore_index=True
    )

    print("Concatenating call data...")
    calls_df = pd.concat(
        [raw_tables['mastercalls_jan_feb'], raw_tables['mastercalls_mar_apr'],
         raw_tables['mastercalls_may_jul'], raw_tables['mastercalls_aug_oct']],
        ignore_index=True
    )

    masterdatagcs_df = raw_tables['masterdatagcs']

    print("Raw data loading and concatenation complete.")
    return {
        "clients_df": clients_df,
        "calls_df": calls_df,
        "masterdatagcs_df": masterdatagcs_df
    }

if __name__ == '__main__':
    # Example of how to run this module directly for testing
    # (This would typically be called from run_pipeline.py or a notebook)
    base_dfs = load_and_concatenate_raw_data()
    print(f"Clients DF shape: {base_dfs['clients_df'].shape}")
    print(f"Calls DF shape: {base_dfs['calls_df'].shape}")
    print(f"GCS DF shape: {base_dfs['masterdatagcs_df'].shape}")

    # Optional: Save interim data
    # base_dfs['clients_df'].to_parquet(os.path.join(settings.INTERIM_DATA_DIR, "clients_raw.parquet"))
    # base_dfs['calls_df'].to_parquet(os.path.join(settings.INTERIM_DATA_DIR, "calls_raw.parquet"))
    # base_dfs['masterdatagcs_df'].to_parquet(os.path.join(settings.INTERIM_DATA_DIR, "gcs_raw.parquet"))