# database_handler.py
# Creates and manages the local SQLite database file.

import sqlite3
from typing import List, Dict

DATABASE_FILE = 'ar_lottery_history.db'

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    return conn

def setup_database():
    """Create the results and predictions tables if they don't exist."""
    conn = create_connection()
    try:
        cursor = conn.cursor()
        # Table 1: Historical Results (Primary data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                issue TEXT PRIMARY KEY, 
                code INTEGER NOT NULL,
                api_timestamp REAL,
                fetch_timestamp REAL
            )
        """)
        # Table 2: AI Predictions (NOW WITH NEW COLUMNS)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                issue TEXT PRIMARY KEY,
                prediction_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_at REAL,
                outcome TEXT, -- Stores WIN/LOSS
                system_confidence_level INTEGER, -- NEW COLUMN for Martingale state
                consecutive_losses INTEGER -- NEW COLUMN for Martingale state
            )
        """)
        conn.commit()
        print(f"[DB] Database setup complete. Data stored in: {DATABASE_FILE}")
    except sqlite3.Error as e:
        print(f"[DB ERROR] Setup failed: {e}")
    finally:
        conn.close()

async def save_result_to_database(data_to_save: List[Dict]) -> bool:
    """Inserts results into the database, handling duplicates."""
    conn = create_connection()
    cursor = conn.cursor()
    success_count = 0
    
    try:
        for record in data_to_save:
            cursor.execute("SELECT issue FROM results WHERE issue=?", (record['issue'],))
            if cursor.fetchone():
                continue 
                
            cursor.execute(
                "INSERT INTO results (issue, code, api_timestamp, fetch_timestamp) VALUES (?, ?, ?, ?)",
                (record['issue'], record['code'], record['api_timestamp'], record['fetch_timestamp'])
            )
            success_count += 1
            
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"[DB ERROR] Save failed: {e}")
        return False
    finally:
        conn.close()
        return success_count > 0

# --- Predictions table helpers ---
def create_predictions_table():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predict_for_issue TEXT,
        predicted_label TEXT,
        confidence REAL,
        engine_details TEXT,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # index for faster lookup
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_predict_for_issue ON predictions(predict_for_issue);")
    conn.commit()
    conn.close()

def save_prediction_to_database(predict_for_issue, predicted_label, confidence, engine_details):
    conn = create_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO predictions (predict_for_issue, predicted_label, confidence, engine_details) VALUES (?, ?, ?, ?)",
            (predict_for_issue, predicted_label, confidence, engine_details)
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] Could not save prediction: {e}")
        return False
    finally:
        conn.close()


# Convenience: ensure DB and tables exist
def ensure_setup():
    setup_database()
    create_predictions_table()
