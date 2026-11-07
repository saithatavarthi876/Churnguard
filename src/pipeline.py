from pathlib import Path
import logging

from src.ingestion import ingest_data
from src.validation import validate_data
from src.transformation import transform_data

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
        validate_data(raw_data)
        cleaned_data = transform_data(raw_data)

        cleaned_data.to_csv(PROCESSED_FILE, index=False)

        logging.info(
            "Pipeline completed. Rows processed: %s",
            len(cleaned_data),
        )
        print(f"Saved processed data to {PROCESSED_FILE}")
        return cleaned_data

    except Exception:
        logging.exception("Pipeline failed.")
        raise


if __name__ == "__main__":
    run_pipeline()