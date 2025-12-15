import os
import pandas as pd # For pandas options, if you want them globally set here
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Project Root ---
# This helps in constructing absolute paths reliably
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root

# --- Database Credentials & Configuration ---
DB_HOST = os.getenv("DB_HOST", "172.20.20.4")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "nos")
DB_USER = os.getenv("DB_USER", "dvolin")
DB_PASSWORD = os.getenv("DB_PASSWORD") # Default if not in .env
DB_SCHEMA = "nos2425"

# --- Table Names ---
# Raw tables to be loaded from the database
RAW_TABLE_NAMES = [
    'masterdataclients_jan_may',
    'masterdataclients_jun_oct',
    'mastercalls_jan_feb',
    'mastercalls_mar_apr',
    'mastercalls_may_jul',
    'mastercalls_aug_oct',
    'masterdatagcs'
]

# Specific table names for concatenation logic
MASTER_CLIENT_TABLES_FOR_CONCAT = ['masterdataclients_jan_may', 'masterdataclients_jun_oct']
MASTER_CALL_TABLES_FOR_CONCAT = [
    'mastercalls_jan_feb', 'mastercalls_mar_apr',
    'mastercalls_may_jul', 'mastercalls_aug_oct'
]
MASTER_GCS_TABLE_NAME = 'masterdatagcs'


# --- Column Definitions & Lists (can also be in a separate column_definitions.py) ---

# GC Cleaner related
GC_COLUMNS_TO_DROP_INITIAL = ['RESOURCE_ID'] # The non-key one
GC_FTR_COLUMNS_PATTERN = 'FTR' # To identify FTR columns for inversion

# For gcs_unique creation
GCS_UNIQUE_AGG_COLUMNS_TO_KEEP = [ # Columns to keep for aggregation in gcs_unique (excluding RESOURCE_KEY and date)
    "COUNT_CALLS", "DAYS_ACTIVE",
    "MEAN_RPC", "MEAN_FTR", "MEAN_TMC", "TOTAL_OTS", "OTS_BY_CALL",
    "COUNT_CALLS_IF", "MEAN_RPC_IF", "MEAN_FTR_IF", "MEAN_TMC_IF", "TOTAL_OTS_IF", "OTS_BY_CALL_IF",
    "COUNT_CALLS_OTHER", "MEAN_RPC_OTHER", "MEAN_FTR_OTHER", "MEAN_TMC_OTHER", "TOTAL_OTS_OTHER", "OTS_BY_CALL_OTHER",
    "COUNT_CALLS_VF", "MEAN_RPC_VF", "MEAN_FTR_VF", "MEAN_TMC_VF", "TOTAL_OTS_VF", "OTS_BY_CALL_VF",
    "COUNT_CALLS_TV", "MEAN_RPC_TV", "MEAN_FTR_TV", "MEAN_TMC_TV", "TOTAL_OTS_TV", "OTS_BY_CALL_TV",
    "COUNT_CALLS_MOVEL", "MEAN_RPC_MOVEL", "MEAN_FTR_MOVEL", "MEAN_TMC_MOVEL", "TOTAL_OTS_MOVEL", "OTS_BY_CALL_MOVEL"
]
GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS = { # Used to check if a category has any calls
    "IF": ["COUNT_CALLS_IF"], # Example, use relevant columns from gcs_unique_df
    "OTHER": ["COUNT_CALLS_OTHER"],
    "VF": ["COUNT_CALLS_VF"],
    "TV": ["COUNT_CALLS_TV"],
    "MOVEL": ["COUNT_CALLS_MOVEL"]
}

# For gcs_detailed creation
GCS_DETAILED_LEG_COUNT_BINARY_FLAG_COLS = [
    "COUNT_LEGS_IF_14_DAYS", "COUNT_LEGS_MOVEL_14_DAYS",
    "COUNT_LEGS_TV_14_DAYS", "COUNT_LEGS_VF_14_DAYS",
    "COUNT_LEGS_IF_30_DAYS", "COUNT_LEGS_MOVEL_30_DAYS",
    "COUNT_LEGS_TV_30_DAYS", "COUNT_LEGS_VF_30_DAYS"
]
GCS_DETAILED_COLUMNS_TO_DROP = ["LEG_START_DATE", "DAYS_ACTIVE", "Column1"]


# Client Cleaner related
# For filling rolling window metrics
CLIENT_METRICS_180_PATTERN = 'MEAN_180' # Or list them explicitly
CLIENT_METRICS_365_PATTERN = 'MEAN_365' # Or list them explicitly
CLIENT_CALL_COUNT_180_TOTAL_COL = 'LEG_IF_ID_COUNT_180_TOTAL'
CLIENT_CALL_COUNT_365_TOTAL_COL = 'LEG_IF_ID_COUNT_365_TOTAL'

# For call count by issue
CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE = [ # {} will be replaced by 180 or 365
    "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_IF",
    "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_MOVEL",
    "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_OTHER",
    "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_TV",
    "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_VF",
    "CALL_IF_ID_NUNIQUE_{}_TOTAL",
]
CLIENT_SUBSCRIPTION_COLS_INDIVIDUAL = ['IF_PROD_FLG', 'TV_PROD_FLG', 'VF_PROD_FLG']
CLIENT_SUBSCRIPTION_COLS_MOBILE = ['VM_PROD_FLG', 'IM_PROD_FLG']
CLIENT_SERVICE_MAP_INDIVIDUAL_180 = {
    'IF_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_180_ASSUNTO_IF',
    'TV_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_180_ASSUNTO_TV',
    'VF_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_180_ASSUNTO_VF',
}
CLIENT_MOBILE_CALL_FLAG_180 = 'Flag_CALL_IF_ID_NUNIQUE_180_ASSUNTO_MOVEL'
CLIENT_SERVICE_MAP_INDIVIDUAL_365 = {
    'IF_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_365_ASSUNTO_IF',
    'TV_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_365_ASSUNTO_TV',
    'VF_PROD_FLG': 'Flag_CALL_IF_ID_NUNIQUE_365_ASSUNTO_VF',
}
CLIENT_MOBILE_CALL_FLAG_365 = 'Flag_CALL_IF_ID_NUNIQUE_365_ASSUNTO_MOVEL'

CLIENT_TIME_FRAMES_FOR_PROP_CALLS = ['180', '365']
CLIENT_PROP_CALLS_SUFFIXES = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF']

CLIENT_CHARACTERISTIC_COLS_FFILL = [
    'IF_DOWNLOAD_SPEED_MBPS_QTY', 'PF_CLIENT_MONTHS', 'NR_SAS',
    'FLG_EMPRESARIAL', 'FLG_CONSUMER', 'FLG_UNKNOWN', 'FLG_NO_BOX',
    'FLG_BOX_3_ULTRA_HD', 'FLG_BOX_2_HD_PLUS_DVR_CABO', 'FLG_BOX_2_HD_CABO',
    'FLG_BOX_1_HD_PLUS_DVR_SAT_TDT', 'FLG_BOX_1_HD_SAT', 'IF_UPLOAD_SPEED_KBPS_QTY',
    'TV_TVCINE_FLG', 'FLG_INDEFINIDO', 'TV_11SPORTS_FLG', 'TV_BTV_FLG', 'TV_SPTV_FLG',
    'TECH_GSM_FLG', 'TECH_DTH_FLG', 'TECH_FTTH_FLG', 'TECH_CABLE_FLG', 'TECH_COPPER_FLG',
    'SERVICES_QTY', 'ARPU_CALCULATED', 'FLG_ILHAS', 'CLIENT_ANTIQUITY_YEARS', 'ARPU_CLIENT'
]
# Columns to create flags for and then fill with -1 if still NaN
CLIENT_COLS_FLAG_AND_FILL_MINUS_ONE = {
    'PF_CLIENT_MONTHS': True, # True means create a "Flag_PF_CLIENT_MONTHS"
    'ARPU_CLIENT': True,
    'CLIENT_ANTIQUITY_YEARS': True,
    'ARPU_CALCULATED': True,
    # Add other characteristic columns from CLIENT_CHARACTERISTIC_COLS_FFILL
    # that should be filled with -1 if they remain NaN after ffill
    'IF_DOWNLOAD_SPEED_MBPS_QTY': False, # Example: fill with -1, but no specific flag
    # ... and so on for the rest of CLIENT_CHARACTERISTIC_COLS_FFILL
}
# This mapping helps to systematically handle the final fill and flag creation.
# Alternatively, just list columns to fill with -1 if the flagging logic is simple.


# --- General Cleaning & Merging Parameters ---
KEY_PERSON_SK = "PERSON_SK"
KEY_RESOURCE_KEY = "RESOURCE_KEY"
DATE_COL_CALLS_RAW = "CALL_START_TIME_DAT" # Original date column in calls_df
DATE_COL_CLIENT_RAW = "START_DATE"         # Original date column in client_df (or LEG_START_TIME_DAY)
NORMALIZED_DATE_COL_CALLS = "CALL_START_TIME_DAT_normalized"
NORMALIZED_DATE_COL_CLIENT = "START_DATE_normalized"


# --- Pandas Display Options (optional, can be set in scripts/notebooks) ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # Can be problematic, often better to set a large number like 1000
pd.set_option('display.max_colwidth', None)


