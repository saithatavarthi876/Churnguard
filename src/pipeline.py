from pathlib import Path
import logging

from src.ingestion import ingest_data
from src.validation import validate_data
from src.transformation import transform_data
from src.database import load_customers, record_pipeline_run

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FILE = ROOT / "data" / "processed" / "customers_clean.csv"
LOG_FILE = ROOT / "logs" / "pipeline.log"


def run_pipeline():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    
    logging.info("Pipeline started.")

    try:
        raw_data = ingest_data()
        logging.info("Ingestion completed.")
        
        validate_data(raw_data)
    
        logging.info("Validation completed.")
        
        cleaned_data = transform_data(raw_data)
        logging.info("Transformation completed.")
        
        cleaned_data.to_csv(PROCESSED_FILE, index=False)
        logging.info("Processed CSV exported.")

        load_customers(cleaned_data)
        logging.info("SQLite customer table updated.")
        
        record_pipeline_run("SUCCESS", len(cleaned_data))
        logging.info("Pipeline completed successfully.")
       
        return cleaned_data
        
    except Exception as error:
        logging.exception("Pipeline failed.")

        try:
            record_pipeline_run(
                "FAILED",
                error_message=str(error),
            )
        except Exception:
            logging.exception("Could not record failed run.")

        raise
    
if __name__ == "__main__":
    run_pipeline()