import sys
import logging
import duckdb
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(r"D:\battery_health")
OUTPUT_DIR = BASE_DIR / "exploration_data"

def generate_master_csvs():
    # Use 4 threads to balance CPU speed without starving Windows I/O
    con = duckdb.connect(config={'threads': 4})

    datasets = ['battery_dataset1', 'battery_dataset2', 'battery_dataset3']

    for ds in datasets:
        ds_out_dir = OUTPUT_DIR / ds
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Explicitly target partitions to prevent wildcard I/O crashing
        meta_path = f"{BASE_DIR.as_posix()}/processed_parquet/metadata/dataset={ds}/*.parquet"
        ts_path = f"{BASE_DIR.as_posix()}/processed_parquet/time_series/dataset={ds}/*.parquet"
        
        logging.info(f"--- Querying distinct vehicles for {ds} ---")
        
        try:
            cars_df = con.execute(f"""
                SELECT DISTINCT car_id 
                FROM read_parquet('{meta_path}')
                WHERE car_id != -1
            """).df()
        except Exception as e:
            logging.error(f"Could not read metadata for {ds}: {e}")
            continue

        if cars_df.empty:
            logging.warning(f"No valid car IDs found for {ds}. Skipping.")
            continue
            
        total_cars = len(cars_df)
        logging.info(f"Found {total_cars} vehicles in {ds}. Beginning CSV generation...")

        for index, row in cars_df.iterrows():
            car_id = int(row['car_id'])
            file_name = f"car_{car_id}_master.csv"
            output_path = (ds_out_dir / file_name).as_posix()

            if Path(output_path).exists():
                logging.info(f"[{index+1}/{total_cars}] Skipping {file_name} (Already exists)")
                continue

            try:
                # The crucial ORDER BY fix to ensure chronological reading
                con.execute(f"""
                    COPY (
                        SELECT 
                            t.timestamp,
                            m.charge_segment,
                            t.step_index,
                            t.v_avg, t.i_charge, t.t_max, t.v_max, t.v_min, t.t_min, t.soc,
                            m.mileage, m.fault_label, m.snippet_id
                        FROM read_parquet('{meta_path}') m
                        JOIN read_parquet('{ts_path}') t ON m.snippet_id = t.snippet_id
                        WHERE m.car_id = {car_id}
                        ORDER BY m.charge_segment ASC, cast(m.snippet_id as INT) ASC, t.step_index ASC
                    ) TO '{output_path}' (HEADER, DELIMITER ',');
                """)
                logging.info(f"[{index+1}/{total_cars}] Successfully generated {file_name}")
            except Exception as e:
                logging.error(f"Failed to generate CSV for Car {car_id}: {str(e)}")

if __name__ == "__main__":
    generate_master_csvs()