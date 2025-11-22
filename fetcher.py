# fetcher.py
# Runs 24/7 to automatically fetch and store data from the API
# NOW POLLING EVERY 20 SECONDS

import requests
import time
import json
from typing import List, Dict
import asyncio
from database_handler import save_result_to_database, ensure_setup, save_prediction_to_database
# OLD ENGINE REMOVED - WE USE prediction_engine ONLY
# from mrperfect_v6_engine import generateFullPredictionFromLabels, initState
import sqlite3



# --- Configuration (FIXED INTERVAL) ---
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
FETCH_INTERVAL_SECONDS = 20
MAX_RETRIES = 5
PAGES_TO_FETCH = 50


# --- Data Parsing Logic ---
def extract_relevant_results(raw_json: Dict) -> List[Dict]:
    results_list = []
    data_dict = raw_json.get('data', {})
    data_list = data_dict.get('list', [])

    if not isinstance(data_list, list):
        return []

    for item in data_list:
        if not isinstance(item, dict):
            continue

        try:
            issue = item.get('issueNumber')
            code_str = item.get('number')

            if issue and code_str is not None:
                code_int = int(code_str)

                results_list.append({
                    'issue': str(issue),
                    'code': code_int,
                    'api_timestamp': item.get('openTime'),
                    'fetch_timestamp': time.time()
                })
        except Exception:
            continue

    return results_list


async def fetch_and_process_data():
    payload = {'pageSize': 15, 'pageIndex': 1}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(API_URL, params=payload, timeout=10)
            response.raise_for_status()

            raw_data = response.json()
            historical_data = extract_relevant_results(raw_data)
            
            # --- Evaluate the previous prediction using the new data ---
            if historical_data:
                try:
                    evaluate_and_update_prediction(historical_data)
                except Exception as e:
                    print("Evaluation hook error:", e)
            # ----------------------------------------------------------------

            success = await save_result_to_database(historical_data)

            if success:
                print(f"[FETCHER] New data saved. Interval {FETCH_INTERVAL_SECONDS}s")

                # auto-predict hook after saving
                try:
                    __run_fetcher_prediction_hook()
                except Exception as e:
                    print("Prediction hook error:", e)

            return

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
            else:
                print("Max retries reached.")
        except Exception as e:
            print("Unexpected error:", e)
            break


def start_fetcher():
    print("----------------------------------------------------------")
    print(f"  Starting 24/7 Data Collector. Polling every {FETCH_INTERVAL_SECONDS} seconds.")
    print("----------------------------------------------------------")

    while True:
        try:
            asyncio.run(fetch_and_process_data())
        except Exception as e:
            print(f"Main loop critical error: {e}")

        time.sleep(FETCH_INTERVAL_SECONDS)


# ========== PREDICTION HOOK & EVALUATION ==========
try:
    from prediction_engine import ultraAIPredict, get_big_small_from_number
except:
    ultraAIPredict = None
    get_big_small_from_number = None
    print("[fetcher patch] prediction_engine missing")


def ensure_predictions_v2_table():
    conn = sqlite3.connect('ar_lottery_history.db')
    cur = conn.cursor()

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

    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_v2_issue ON predictions_v2(predict_for_issue)")
    conn.commit()
    conn.close()


def save_prediction_v2(predict_for_issue, predicted_label, confidence, engine_details, data_store_count=None):
    try:
        conn = sqlite3.connect('ar_lottery_history.db')
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO predictions_v2 
        (predict_for_issue, predicted_label, confidence, engine_details, data_store_count)
        VALUES (?, ?, ?, ?, ?)
        """, (str(predict_for_issue), predicted_label, float(confidence), json.dumps(engine_details), data_store_count))

        conn.commit()
    except Exception as e:
        print("Prediction save error:", e)
    finally:
        conn.close()


def evaluate_and_update_prediction(new_results: List[Dict]):
    """Checks if any new result evaluates an existing, unevaluated prediction."""
    if ultraAIPredict is None or get_big_small_from_number is None or not new_results:
        return

    conn = sqlite3.connect('ar_lottery_history.db')
    cur = conn.cursor()

    try:
        # We focus on the most recently fetched result
        latest_result = new_results[0]
        actual_issue = latest_result.get('issue')
        actual_code = latest_result.get('code')
        
        if not actual_issue or actual_code is None:
            return

        # 1. Calculate the actual label (BIG or SMALL)
        try:
            # Extracts the last digit of the code
            num = int(str(actual_code)[-1])
            actual_label = get_big_small_from_number(num)
        except Exception:
            actual_label = None
            print(f"Error calculating label for issue {actual_issue}")
            return

        # 2. Find the prediction for this issue that needs evaluation
        cur.execute("""
            SELECT id, predicted_label FROM predictions_v2 
            WHERE predict_for_issue = ? AND actual_label IS NULL
        """, (actual_issue,))
        
        prediction_record = cur.fetchone()

        if prediction_record:
            pred_id, predicted_label = prediction_record
            
            # 3. Evaluate the prediction
            is_win = 0
            win_loss_status = "LOSS"
            if actual_label is not None and predicted_label == actual_label:
                is_win = 1
                win_loss_status = "WIN"
                
            # 4. Update the prediction record
            cur.execute("""
                UPDATE predictions_v2 SET 
                actual_label = ?, 
                is_win = ?, 
                evaluated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (actual_label, is_win, pred_id))
            
            conn.commit()
            print(f"[EVALUATOR] Issue {actual_issue}: Predicted {predicted_label}, Actual {actual_label} -> {win_loss_status}")

    except Exception as e:
        print("evaluate_and_update_prediction error:", e)
    finally:
        conn.close()


def run_prediction_hook():
    if ultraAIPredict is None:
        return

    try:
        conn = sqlite3.connect('ar_lottery_history.db')
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(results)")
        cols = [r[1] for r in cur.fetchall()]

        cur.execute("SELECT * FROM results ORDER BY rowid ASC")
        rows = cur.fetchall()
        data_store_count = len(rows)

        history = []
        for r in rows:
            rowd = dict(zip(cols, r))

            entry = {
                'issue': rowd.get('issue'),
                'code': rowd.get('code')
            }

            try:
                # Extracts the last digit of the code
                num = int(str(entry['code'])[-1])
            except:
                # If extraction fails (e.g., code is not a valid number), num is None
                num = None

            entry['actual_number'] = num
            entry['actual_outcome'] = None # Default the outcome to None

            # ONLY call get_big_small_from_number if 'num' is successfully extracted (is not None)
            if num is not None and get_big_small_from_number:
                entry['actual_outcome'] = get_big_small_from_number(num)
            
            # ------------------ FIX APPLIED HERE ------------------
            # Only append the entry if we successfully got an outcome.
            # This prevents passing 'actual_outcome': None to ultraAIPredict.
            if entry['actual_outcome'] is not None:
                history.append(entry)
            # ------------------------------------------------------


        if not history:
            print("[AUTO-PREDICT] No valid history to process.")
            return

        engine_input = list(reversed(history))
        out = ultraAIPredict(engine_input, None)

        predicted = out.get('finalDecision') or out.get('prediction') or 'N/A'
        confidence = out.get('finalConfidence') or 0.5

        next_issue = str(int(history[-1]['issue']) + 1)

        ensure_predictions_v2_table()
        save_prediction_v2(next_issue, predicted, confidence, out, data_store_count)

        print(f"[AUTO-PREDICT] {next_issue} -> {predicted} | conf {confidence} | stored {data_store_count}")

    except Exception as e:
        print("run_prediction_hook error:", e)


def __run_fetcher_prediction_hook():
    try:
        run_prediction_hook()
    except Exception as e:
        print("Hook failure:", e)


if __name__ == '__main__':
    start_fetcher()