
#!/usr/bin/env python3
"""
live_predictor.py

Realtime predictor that:
- watches the SQLite DB (ar_lottery_history.db) for new results
- when a new result is found, records it and (optionally) updates model weights if supported
- continuously emits a prediction for the next period using prediction_engine.ultraAIPredict
- saves predictions into predictions_v2 table to avoid schema conflict with existing tables
"""

import sqlite3, time, json, sys, os, traceback
from datetime import datetime
from prediction_engine import ultraAIPredict, get_big_small_from_number


def _parse_issue_to_int(issue_str):
    """Try to extract a trailing integer from issue string; return int or None."""
    if issue_str is None:
        return None
    s = str(issue_str)
    # Try full numeric
    if s.isdigit():
        try:
            return int(s)
        except:
            return None
    # Try to find the longest contiguous numeric substring (prefer trailing)
    import re
    parts = re.findall(r'\d+', s)
    if not parts:
        return None
    # Prefer last numeric group
    try:
        return int(parts[-1])
    except:
        return None



def _build_shared_stats(conn, history):
    """
    Build a shared_stats_payload dict expected by ultraAIPredict to avoid None streaks.
    Uses latest history and predictions_v2 table to compute last outcomes and streaks.
    """
    payload = {'consecutiveSkips':0, 'consecutiveLosses':0, 'consecutiveWins':0, 'consecutiveSamePredictions':0}
    try:
        cur = conn.cursor()
        # last actual outcome
        last_actual = history[-1].get('actual_outcome') if history else None
        payload['lastActualOutcome'] = last_actual
        # find last prediction for that issue
        last_issue = history[-1].get('issue') if history else None
        if last_issue:
            cur.execute("SELECT predicted_label, is_win FROM predictions_v2 WHERE predict_for_issue = ? ORDER BY id DESC LIMIT 1", (str(last_issue),))
            row = cur.fetchone()
            if row:
                payload['lastPredictedOutcome'] = row[0]
                # set counts based on is_win
                payload['lastPredictionWasWin'] = bool(row[1])
        # compute consecutive wins/losses from recent evaluated predictions
        cur.execute("SELECT is_win FROM predictions_v2 WHERE is_win IS NOT NULL ORDER BY id DESC LIMIT 20")
        rows = cur.fetchall()
        cons_wins = cons_losses = 0
        for r in rows:
            v = r[0]
            if v is None:
                break
            if v == 1:
                if cons_losses==0:
                    cons_wins += 1
                else:
                    break
            else:
                if cons_wins==0:
                    cons_losses += 1
                else:
                    break
        payload['consecutiveWins'] = cons_wins
        payload['consecutiveLosses'] = cons_losses
    except Exception:
        pass
    return payload

def _prediction_exists(conn, predict_for_issue):
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(1) FROM predictions_v2 WHERE predict_for_issue = ?", (str(predict_for_issue),))
        r = cur.fetchone()
        return (r and r[0] and r[0] > 0)
    except Exception:
        # Table might not exist or different schema; attempt safe check
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions_v2'")
            if cur.fetchone() is None:
                return False
        except:
            return False
    return False


DB = "ar_lottery_history.db"
POLL_INTERVAL = 8  # seconds

def ensure_predictions_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predict_for_issue TEXT,
        predicted_label TEXT,
        confidence REAL,
        engine_details TEXT,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_v2_issue ON predictions_v2(predict_for_issue)")
    # Ensure new columns for evaluation exist
    try:
        cur.execute("ALTER TABLE predictions_v2 ADD COLUMN actual_label TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE predictions_v2 ADD COLUMN is_win INTEGER")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE predictions_v2 ADD COLUMN evaluated_at TIMESTAMP")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE predictions_v2 ADD COLUMN data_store_count INTEGER")
    except Exception:
        pass

    conn.commit()

def extract_last_digit_from_code(code):
    if code is None:
        return None
    s = str(code)
    parts = [p for p in s.replace('\\n','').split(',') if any(ch.isdigit() for ch in p)]
    if not parts:
        digits = ''.join(ch for ch in s if ch.isdigit())
        if not digits: return None
        return int(digits[-1])
    last = parts[-1].strip()
    digits = ''.join(ch for ch in last if ch.isdigit())
    if not digits: return None
    return int(digits[-1])

def load_history(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(results)")
    cols = [r[1] for r in cur.fetchall()]
    cur.execute("SELECT * FROM results ORDER BY rowid ASC")
    rows = cur.fetchall()
    history = []
    for r in rows:
        rowd = dict(zip(cols, r))
        entry = {}
        entry['rowid'] = rowd.get('rowid') or rowd.get('id') or None
        entry['issue'] = rowd.get('issue') or rowd.get('issue_no') or rowd.get('issueNumber') or rowd.get('issue_id') or rowd.get('issueCode') or str(entry['rowid'])
        entry['code'] = rowd.get('code') or rowd.get('value') or rowd.get('numbers') or ''
        entry['api_timestamp'] = rowd.get('api_timestamp') or rowd.get('timestamp') or None
        entry['fetch_timestamp'] = rowd.get('fetch_timestamp') or rowd.get('fetched_at') or None
        num = extract_last_digit_from_code(entry['code'])
        entry['actual_number'] = num
        entry['actual_outcome'] = get_big_small_from_number(num) if num is not None else None
        history.append(entry)
    return history

def save_prediction(conn, predict_for_issue, predicted_label, confidence, engine_details, data_store_count=None):
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions_v2 (predict_for_issue, predicted_label, confidence, engine_details, data_store_count) VALUES (?, ?, ?, ?, ?)", 
                (str(predict_for_issue), predicted_label, float(confidence), json.dumps(engine_details), data_store_count))
    conn.commit()

def main_loop():
    print("Starting live predictor. DB:", DB)
    conn = sqlite3.connect(DB, timeout=30)
    ensure_predictions_table(conn)
    last_row_count = 0
    # initialize last_row_count
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM results")
    try:
        last_row_count = cur.fetchone()[0]
    except Exception:
        last_row_count = 0

    while True:
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM results")
            row_count = cur.fetchone()[0]
            if row_count != last_row_count:
                print(f"[{datetime.now()}] New results detected: {last_row_count} -> {row_count}")

                # --- Evaluate previous prediction (if any) for the issue that just arrived ---
                try:
                    current_issue = history[-1].get('issue') or ''
                    actual = history[-1].get('actual_outcome') or None
                    if current_issue and actual is not None:
                        # Try to find a prediction saved for this issue
                        cur.execute("SELECT id, predicted_label FROM predictions_v2 WHERE predict_for_issue = ?", (str(current_issue),))
                        row = cur.fetchone()
                        if row:
                            pred_id, pred_label = row[0], row[1]
                            is_win = 1 if (pred_label == actual) else 0
                            cur.execute("UPDATE predictions_v2 SET actual_label = ?, is_win = ?, evaluated_at = CURRENT_TIMESTAMP WHERE id = ?", (str(actual), is_win, pred_id))
                            conn.commit()
                            print(f"[{datetime.now()}] EVALUATION -> Issue:{current_issue} Pred:{pred_label} Actual:{actual} -> {'WIN' if is_win==1 else 'LOSS'}")
                except Exception as e:
                    print('Evaluation error:', e)

                # load full history and compute prediction for next
                history = load_history(conn)
                if not history:
                    last_row_count = row_count
                    time.sleep(POLL_INTERVAL)
                    continue
                # call engine to generate prediction for next period using history (newest-first expected)
                try:
                    # engine expects newest-first order, so reverse history list
                    engine_input = list(reversed(history))
                    out = ultraAIPredict(engine_input, None)
                except Exception as e:
                    print("Engine error on prediction call:", e)
                    traceback.print_exc()
                    out = {}
                predicted = out.get('finalDecision') or out.get('lastPredictedOutcome') or out.get('finalDecision') or out.get('prediction') or out.get('predicted') or 'N/A'
                confidence = out.get('finalConfidence') or out.get('lastFinalConfidence') or out.get('confidence') or 0.5
                next_issue_est = f"after_{history[-1].get('issue','')}"  # placeholder tag for next issue
                print(f"[{datetime.now()}] LIVE PREDICTION -> Pred:{predicted} | Conf:{confidence:.2f} | NextIssue:{next_issue_est} | data_store_count:{data_store_count}")
                # save to predictions_v2
                try:
                    save_prediction(conn, next_issue_est, predicted, confidence, out)
                except Exception as e:
                    print("Save prediction error:", e)
                # After saving, also attempt to evaluate if prior saved prediction had been for the last issue and update weights if possible.
                # (prediction_engine may or may not expose weight-update. We skip weight update unless engine provides a safe API.)
                last_row_count = row_count
            else:
                # Produce a heartbeat prediction without saving (optional)
                cur = conn.cursor()
                cur.execute("SELECT * FROM results ORDER BY rowid DESC LIMIT 1")
                last = cur.fetchone()
                if last:
                    history = load_history(conn)
                    data_store_count = len(history)
                    try:
                        shared_stats = _build_shared_stats(conn, history)
                        out = ultraAIPredict(list(reversed(history)), shared_stats)
                        pred = out.get('finalDecision') or out.get('prediction') or out.get('predicted') or 'N/A'
                        conf = out.get('finalConfidence') or out.get('confidence') or 0.5
                        last_issue_val = history[-1].get('issue')
                    last_issue_int = _parse_issue_to_int(last_issue_val)
                    running_period = None
                    if last_issue_int is not None:
                        try:
                            running_period = last_issue_int + 1
                        except:
                            running_period = None
                    if running_period is not None:
                        print(f"[{datetime.now()}] HEARTBEAT Pred:{pred} | Conf:{conf:.2f} | last_issue:{last_issue_val} | running_period:{running_period}")
                    else:
                        print(f"[{datetime.now()}] HEARTBEAT Pred:{pred} | Conf:{conf:.2f} | last_issue:{last_issue_val}")
                    except Exception as e:
                        print("Heartbeat engine error:", e)
                # else: no rows yet
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("Stopping live predictor.")
            break
        except Exception as e:
            print("Main loop exception:", e)
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()
