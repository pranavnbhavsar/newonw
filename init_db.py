import sqlite3
import os

DB_FILE = "ar_lottery_history.db"

def initialize_database():
    print(f"Initializing {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # 1. Create 'results' table (Required by fetcher.py)
    print("Creating table: results")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            issue TEXT PRIMARY KEY, 
            code INTEGER NOT NULL,
            api_timestamp REAL,
            fetch_timestamp REAL
        )
    """)

    # 2. Create 'predictions_v2' table (Required by fetcher.py and live_predictor.py)
    print("Creating table: predictions_v2")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predict_for_issue TEXT,
        predicted_label TEXT,
        confidence REAL,
        engine_details TEXT,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        actual_label TEXT,
        is_win INTEGER,
        evaluated_at TIMESTAMP,
        data_store_count INTEGER
    )
    """)
    
    # Create index for faster lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_v2_issue ON predictions_v2(predict_for_issue)")

    conn.commit()
    conn.close()
    print("Database setup complete!")

if __name__ == "__main__":
    initialize_database()