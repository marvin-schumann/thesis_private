import sys
import os
import logging
import pandas as pd

# Ensure the project root is in PYTHONPATH to find 'config' and 'libPBL2425NovaNOS'
PROJECT_ROOT_FROM_RUNNER = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT_FROM_RUNNER not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FROM_RUNNER)

from config import settings
from libPBL2425NovaNOS.data_access import loader
from libPBL2425NovaNOS.data_cleaning import call_cleaner, client_cleaner, gc_cleaner, merger
from libPBL2425NovaNOS.modelling import modelling

# --- Logging Setup ---
# Configure logging once for the entire application run
# Ensure LOGS_DIR exists (settings.py creates it, but double-check or create here)
os.makedirs(settings.LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=settings.LOG_LEVEL,  # e.g., "INFO" or "DEBUG"
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE, mode='w'),  # Overwrite log file each run
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)
logger = logging.getLogger(__name__) # Get a logger for this main script

def main_pipeline():
    logger.info("====== Starting NOS Modelling Pipeline ======")

    # --- 1. Data Access ---
    logger.info("--- Stage 1: Data Access ---")
    try:
        raw_data_dict = loader.load_and_concatenate_raw_data()
        clients_raw_df = raw_data_dict['clients_df']
        calls_raw_df = raw_data_dict['calls_df']
        gcs_raw_df = raw_data_dict['gcs_df']
        
        if clients_raw_df.empty or calls_raw_df.empty or gcs_raw_df.empty:
            logger.error("One or more raw dataframes are empty after loading. Aborting.")
            return
    except Exception as e:
        logger.error(f"Error in Data Access stage: {e}", exc_info=True)
        return # Stop pipeline if data loading fails

    # --- 2. Data Cleaning ---
    logger.info("--- Stage 2: Data Cleaning ---")
    try:
        logger.info("Cleaning call data...")
        calls_cleaned_df = call_cleaner.initial_call_cleaning(calls_raw_df)
        
        logger.info("Cleaning client data...")
        client_cleaned_df = client_cleaner.full_client_cleaning(clients_raw_df)
        
        logger.info("Cleaning GCS data...")
        gcs_unique_cleaned_df = gc_cleaner.create_gcs_unique_with_mode_aggregation(gcs_raw_df)

        if calls_cleaned_df.empty or client_cleaned_df.empty or gcs_unique_cleaned_df.empty:
            logger.warning("One or more dataframes are empty after cleaning. This may affect merging and modelling.")
            # Decide if to abort based on which DF is empty
            if calls_cleaned_df.empty : 
                logger.error("calls_cleaned_df is empty. Aborting.")
                return
    except Exception as e:
        logger.error(f"Error in Data Cleaning stage: {e}", exc_info=True)
        return

    # --- 3. Data Merging ---
    logger.info("--- Stage 3: Data Merging ---")
    try:
        final_modelling_df = merger.create_modeling_dataset(
            calls_cleaned_df=calls_cleaned_df,
            client_cleaned_df=client_cleaned_df,
            gcs_unique_df=gcs_unique_cleaned_df
        )
        if final_modelling_df.empty:
            logger.error("Merged dataset for modelling is empty. Aborting.")
            return
    except Exception as e:
        logger.error(f"Error in Data Merging stage: {e}", exc_info=True)
        return

    # --- 4. Modelling ---
    logger.info("--- Stage 4: Modelling ---")
    try:
        # Pass both the merged dataset and the cleaned gcs_unique for the simulation part
        modelling_results = modelling.run_modelling(
            df_input=final_modelling_df,
            gcs_unique_input=gcs_unique_cleaned_df # Pass the cleaned GCS data
        )
        
        if 'error' in modelling_results:
            logger.error(f"Modelling stage failed: {modelling_results['error']}")
            return

        logger.info("Modelling Results Summary:")
        if 'tmc' in modelling_results and 'metrics' in modelling_results['tmc']:
            logger.info(f"TMC Model Metrics: {modelling_results['tmc']['metrics']}")
        if 'ftr' in modelling_results and 'metrics' in modelling_results['ftr']:
            logger.info(f"FTR Model Metrics: {modelling_results['ftr']['metrics']}")
        if 'ot' in modelling_results and 'metrics' in modelling_results['ot']:
            logger.info(f"OT Model Metrics: {modelling_results['ot']['metrics']}")

        if 'simulation_df' in modelling_results:
            simulation_output_df = modelling_results['simulation_df']
            logger.info(f"Simulation produced {len(simulation_output_df)} recommendations.")
            if not simulation_output_df.empty:
                logger.info("Sample of simulation output (first 5 rows):")
                # Convert DataFrame to string for clean logging without pandas display options
                try:
                    from io import StringIO
                    output_buffer = StringIO()
                    simulation_output_df.head().to_string(buf=output_buffer)
                    logger.info("\n" + output_buffer.getvalue())
                except ImportError: # Fallback if io not available or issue with to_string
                    for _, row in simulation_output_df.head().iterrows():
                        logger.info(row.to_dict())


        # At this point, modelling_results dictionary contains trained models, scalers, metrics, etc.
        # These are "stored internally". If you need to save them, do it here.
        # e.g., using joblib or pickle for models/scalers.
        # Example: joblib.dump(modelling_results['tmc']['model'], 'tmc_model.joblib')

    except Exception as e:
        logger.error(f"Error in Modelling stage: {e}", exc_info=True)
        return

    logger.info("====== NOS Modelling Pipeline Finished Successfully ======")

if __name__ == "__main__":
    main_pipeline()