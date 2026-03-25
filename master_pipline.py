import os
import sys
import logging
import duckdb
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Lock torch to CPU to prevent CUDA memory leaks across workers
torch.cuda.is_available = lambda: False 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- DIRECTORIES ---
BASE_DIR = Path(r"D:\battery_health")
RAW_DATA_DIR = BASE_DIR / "raw_data"
PARQUET_DIR = BASE_DIR / "processed_parquet"
CSV_DIR = BASE_DIR / "exploration_data"

CHUNK_SIZE = 100 
TS_COLUMNS = [
    "v_avg", "i_charge", "soc", "v_max", 
    "v_min", "t_max", "t_min", "timestamp"
]

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_dataset_labels(dataset_name):
    """Reads the label.csv for a specific dataset and returns a dictionary mapping car_id -> label"""
    label_dir = RAW_DATA_DIR / dataset_name / "label"
    label_dict = {}
    
    if not label_dir.exists():
        logging.warning(f"No label directory found for {dataset_name}.")
        return label_dict

    try:
        # Grab the first file in the label directory
        label_file = next(label_dir.glob("*.*"))
        df = pd.read_csv(label_file)
        
        # Ensure it has 'car' and 'label' columns
        if 'car' in df.columns and 'label' in df.columns:
            # Convert to dictionary: {car_id: label}
            label_dict = pd.Series(df['label'].values, index=df['car'].values).to_dict()
            logging.info(f"Loaded {len(label_dict)} vehicle labels for {dataset_name}.")
        else:
            logging.warning(f"Label file {label_file.name} missing 'car' or 'label' columns.")
    except StopIteration:
        logging.warning(f"Label directory for {dataset_name} is empty.")
    except Exception as e:
        logging.error(f"Failed to read labels for {dataset_name}: {e}")
        
    return label_dict

def process_chunk(chunk_paths, dataset_name, chunk_index, labels_dict):
    """The Worker function that unpickles data and injects labels simultaneously"""
    import gc
    
    meta_part_dir = PARQUET_DIR / "metadata" / f"dataset={dataset_name}"
    ts_part_dir = PARQUET_DIR / "time_series" / f"dataset={dataset_name}"
    
    meta_file = meta_part_dir / f"chunk_{chunk_index:06d}.parquet"
    ts_file = ts_part_dir / f"chunk_{chunk_index:06d}.parquet"

    meta_buffer = []
    ts_buffer = []
    
    for file_path in chunk_paths:
        file_path_obj = Path(file_path)
        
        try:
            snippet_id = int(file_path_obj.stem) 
            data_tuple = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            
            ts_data = data_tuple[0]
            metadata = data_tuple[1]
            
            car_id = int(metadata.get("car", -1))
            
            # INJECT LABEL HERE: Look up the car_id in the dictionary we passed in. Default to -1 if missing.
            fault_label = labels_dict.get(car_id, -1)
            
            meta_record = {
                "snippet_id": snippet_id,
                "car_id": car_id,
                "dataset": dataset_name,
                "charge_segment": int(metadata.get("charge_segment", -1)),
                "mileage": float(metadata.get("mileage", 0.0)),
                "capacity": float(metadata.get("capacity", 0.0)),
                "fault_label": int(fault_label) 
            }
            meta_buffer.append(meta_record)
            
            # Convert Time Series
            ts_array = ts_data.numpy() if isinstance(ts_data, torch.Tensor) else np.array(ts_data)
            steps = ts_array.shape[0]
            
            ts_df_chunk = pd.DataFrame(ts_array, columns=TS_COLUMNS)
            ts_df_chunk.insert(0, "step_index", range(steps))
            ts_df_chunk.insert(0, "snippet_id", snippet_id)
            ts_df_chunk.insert(0, "dataset", dataset_name)
            ts_buffer.append(ts_df_chunk)
            
            del data_tuple, ts_data
            
        except Exception as e:
            continue

    if not meta_buffer or not ts_buffer:
        return f"[{dataset_name}] Chunk {chunk_index} empty after error filtering."

    # Write Parquet Files
    meta_df = pd.DataFrame(meta_buffer)
    ts_df = pd.concat(ts_buffer, ignore_index=True)
    
    pq.write_table(pa.Table.from_pandas(meta_df), meta_file)
    pq.write_table(pa.Table.from_pandas(ts_df), ts_file)
    
    del meta_df, ts_df, meta_buffer, ts_buffer
    gc.collect()
    
    return f"[{dataset_name}] Processed chunk {chunk_index}"

def run_ingestion(dataset_name):
    """Handles the multiprocessing execution for a specific dataset"""
    dataset_path = RAW_DATA_DIR / dataset_name / "data"
    if not dataset_path.exists():
        logging.warning(f"Raw data path not found: {dataset_path}")
        return False

    # Pre-create Parquet directories
    (PARQUET_DIR / "metadata" / f"dataset={dataset_name}").mkdir(parents=True, exist_ok=True)
    (PARQUET_DIR / "time_series" / f"dataset={dataset_name}").mkdir(parents=True, exist_ok=True)

    logging.info(f"--- Scanning {dataset_name} for .pkl files ---")
    all_files = [entry.path for entry in os.scandir(dataset_path) if entry.name.endswith(".pkl") and entry.is_file()]
                
    total_files = len(all_files)
    chunks = list(chunker(all_files, CHUNK_SIZE))
    total_chunks = len(chunks)

    # Load Labels into memory BEFORE starting workers
    labels_dict = get_dataset_labels(dataset_name)

    logging.info(f"Found {total_files} files in {dataset_name}. Spinning up workers...")

    # Protect Windows RAM from thrashing
    worker_count = min(6, max(1, os.cpu_count() - 2))
    
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_chunk, chunk, dataset_name, i, labels_dict): i 
            for i, chunk in enumerate(chunks)
        }
        
        completed_chunks = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_chunks += 1
                if completed_chunks % 50 == 0 or completed_chunks == total_chunks:
                    logging.info(f"{dataset_name} Progress: {completed_chunks}/{total_chunks} chunks")
            except Exception as exc:
                logging.error(f"Chunk generated an exception: {exc}")
                
    return True

def run_csv_generation(dataset_name):
    """Uses DuckDB to dynamically join Parquet and export Master CSVs"""
    logging.info(f"--- Generating Master CSVs for {dataset_name} ---")
    
    ds_out_dir = CSV_DIR / dataset_name
    ds_out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = f"{PARQUET_DIR.as_posix()}/metadata/dataset={dataset_name}/*.parquet"
    ts_path = f"{PARQUET_DIR.as_posix()}/time_series/dataset={dataset_name}/*.parquet"
    
    con = duckdb.connect(config={'threads': 4})
    
    try:
        cars_df = con.execute(f"SELECT DISTINCT car_id FROM read_parquet('{meta_path}') WHERE car_id != -1").df()
    except Exception as e:
        logging.error(f"Failed to read Parquet for CSV generation: {e}")
        return

    total_cars = len(cars_df)
    for index, row in cars_df.iterrows():
        car_id = int(row['car_id'])
        output_path = (ds_out_dir / f"car_{car_id}_master.csv").as_posix()

        try:
            con.execute(f"""
                COPY (
                    SELECT 
                        t.timestamp, m.charge_segment, t.step_index,
                        t.v_avg, t.i_charge, t.soc, t.v_max, t.v_min, t.t_max, t.t_min,
                        m.mileage, m.fault_label, m.snippet_id
                    FROM read_parquet('{meta_path}') m
                    JOIN read_parquet('{ts_path}') t ON m.snippet_id = t.snippet_id
                    WHERE m.car_id = {car_id}
                    ORDER BY m.charge_segment ASC, cast(m.snippet_id as INT) ASC, t.step_index ASC
                ) TO '{output_path}' (HEADER, DELIMITER ',');
            """)
            logging.info(f"[{dataset_name}] Exported car_{car_id}_master.csv ({index+1}/{total_cars})")
        except Exception as e:
            logging.error(f"Failed to generate CSV for Car {car_id}: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    logging.info("=========================================")
    logging.info("   STARTING END-TO-END MASTER PIPELINE   ")
    logging.info("=========================================")
    
    datasets = ["battery_dataset1", "battery_dataset2", "battery_dataset3"]
    
    for ds in datasets:
        success = run_ingestion(ds)
        if success:
            run_csv_generation(ds)
            
    logging.info("=========================================")
    logging.info("        PIPELINE FULLY COMPLETE          ")
    logging.info("=========================================")