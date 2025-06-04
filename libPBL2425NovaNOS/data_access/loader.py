# libPBL2425NovaNOS/data_access/loader.py
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Dict, Any
import sys
import os
import logging

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings

logger = logging.getLogger(__name__)

def get_db_engine():
    """Creates and returns a SQLAlchemy engine."""
    if settings.DB_PASSWORD is None:
        logger.error("DB_PASSWORD is not set. Please check your .env file or environment variables.")
        raise ValueError("DB_PASSWORD not configured.")
    
    engine_url = (
        f'postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}@'
        f'{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}'
    )
    try:
        engine = create_engine(engine_url)
        # Test connection
        with engine.connect() as connection:
            logger.info("Database engine created and connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine or connect: {e}")
        raise

def fetch_table_from_db(table_name: str, engine) -> pd.DataFrame:
    """Fetches a complete table from the database."""
    query = f"SELECT * FROM {settings.DB_SCHEMA}.{table_name};"
    try:
        df = pd.read_sql_query(query, engine)
        logger.info(f"Successfully fetched table: {table_name}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error fetching table {table_name}: {e}")
        raise

def load_and_concatenate_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Loads all raw tables from the database, as defined in settings,
    and concatenates client and call tables.
    Returns a dictionary of the three base DataFrames:
    'clients_df', 'calls_df', 'gcs_df'.
    """
    engine = get_db_engine()
    logger.info("Fetching tables from database...")
    
    raw_tables = {}
    for table_name in settings.RAW_TABLE_NAMES:
        logger.info(f"Fetching {table_name}...")
        raw_tables[table_name] = fetch_table_from_db(table_name, engine)

    logger.info("Concatenating client data...")
    client_dfs_to_concat = [raw_tables[name] for name in settings.MASTER_CLIENT_TABLES_FOR_CONCAT if name in raw_tables]
    if not client_dfs_to_concat:
        logger.warning("No client tables found for concatenation based on settings.MASTER_CLIENT_TABLES_FOR_CONCAT.")
        clients_df = pd.DataFrame()
    else:
        clients_df = pd.concat(client_dfs_to_concat, ignore_index=True)

    logger.info("Concatenating call data...")
    call_dfs_to_concat = [raw_tables[name] for name in settings.MASTER_CALL_TABLES_FOR_CONCAT if name in raw_tables]
    if not call_dfs_to_concat:
        logger.warning("No call tables found for concatenation based on settings.MASTER_CALL_TABLES_FOR_CONCAT.")
        calls_df = pd.DataFrame()
    else:
        calls_df = pd.concat(call_dfs_to_concat, ignore_index=True)
    
    if settings.MASTER_GCS_TABLE_NAME in raw_tables:
        gcs_df = raw_tables[settings.MASTER_GCS_TABLE_NAME]
    else:
        logger.warning(f"Master GCS table '{settings.MASTER_GCS_TABLE_NAME}' not found in fetched tables.")
        gcs_df = pd.DataFrame()

    logger.info("Raw data loading and concatenation complete.")
    logger.info(f"Clients DF shape: {clients_df.shape}")
    logger.info(f"Calls DF shape: {calls_df.shape}")
    logger.info(f"GCS DF shape: {gcs_df.shape}")
    
    return {
        "clients_df": clients_df,
        "calls_df": calls_df,
        "gcs_df": gcs_df  # Renamed from masterdatagcs_df for consistency
    }