# prediction_engine.py
import math
import random
import json
import os
from datetime import datetime

# --- Global State & Configuration ---
signalPerformance = {}
REGIME_SIGNAL_PROFILES = {}
driftDetector = {
    'p_min': float('inf'), 's_min': float('inf'), 'n': 0, 'warning_level': 2.0, 'drift_level': 3.0
}
consecutiveHighConfLosses = 0
reflexiveCorrectionActive = 0
engineMode = "NORMAL"
qTable = {}
GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE = 0.5
heuristicPerformance = {}
systemConsecutiveLosses = 0
shared_stats_payload = {
    'lastPeriodFull': None,
    'lastPredictedOutcome': None,
    'lastActualOutcome': None,
    'lastFinalConfidence': None,
    'lastConfidenceLevel': None,
    'lastMacroRegime': None,
    'lastPredictionSignals': [],
    'lastConcentrationModeEngaged': False,
    'lastMarketEntropyState': None,
    'lastVolatilityRegime': None,
    'longTermGlobalAccuracy': 0.5,
    'consecutiveSamePredictions': 0,
    'consecutiveLosses': 0, # Consolidated key for tracking losses
    'historicalMaxStreaks': {'BIG': 0, 'SMALL': 0},
    'risk_aversion_factor': 1.0,
    'consecutiveSkips': 0
}

# --- New Q-Learning & Engine State ---
learningRate = 0.1
discountFactor = 0.9
explorationRate = 0.1
MIN_DATA_FOR_COMPLEX_AI = 20
ENGINE_PERFORMANCE = {}
ENGINES = ['MasterAI', 'QuantumAI', 'NeuralNet', 'Fibonacci']


# --- Constants for Dynamic Weighting and Learning ---
PERFORMANCE_WINDOW = 50
MIN_OBSERVATIONS_FOR_ADJUST = 8
MAX_WEIGHT_FACTOR = 2.0
MIN_WEIGHT_FACTOR = 0.05
MAX_ALPHA_FACTOR = 1.7
MIN_ALPHA_FACTOR = 0.3
MIN_ABSOLUTE_WEIGHT = 0.0003
ALPHA_UPDATE_RATE = 0.06
PROBATION_THRESHOLD_ACCURACY = 0.40
PROBATION_MIN_OBSERVATIONS = 15
PROBATION_WEIGHT_CAP = 0.10
REGIME_ACCURACY_WINDOW = 60
REGIME_LEARNING_RATE_BASE = 0.028
MAX_CONSECUTIVE_SKIPS = 2


def initialize_state():
    global signalPerformance, REGIME_SIGNAL_PROFILES, driftDetector, consecutiveHighConfLosses, reflexiveCorrectionActive, engineMode, qTable, GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE, heuristicPerformance, systemConsecutiveLosses, shared_stats_payload, learningRate, discountFactor, explorationRate, MIN_DATA_FOR_COMPLEX_AI, ENGINE_PERFORMANCE, ENGINES
    
    signalPerformance = {}
    REGIME_SIGNAL_PROFILES = {
        "TREND_STRONG_LOW_VOL": {"baseWeightMultiplier": 1.40, "activeSignalTypes": ["trend", "momentum", "ichimoku", "volBreak", "leadLag", "stateSpace", "fusion", "ats", "pattern", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.40, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "TREND_STRONG_MED_VOL": {"baseWeightMultiplier": 1.30, "activeSignalTypes": ["trend", "momentum", "ichimoku", "pattern", "leadLag", "stateSpace", "fusion", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.30, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "TREND_STRONG_HIGH_VOL": {"baseWeightMultiplier": 0.80, "activeSignalTypes": ["trend", "ichimoku", "entropy", "volPersist", "zScore", "fusion", "ai", "pattern", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 0.80, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "TREND_MOD_LOW_VOL": {"baseWeightMultiplier": 1.25, "activeSignalTypes": ["trend", "momentum", "ichimoku", "pattern", "volBreak", "leadLag", "stateSpace", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.25, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "TREND_MOD_MED_VOL": {"baseWeightMultiplier": 1.20, "activeSignalTypes": ["trend", "momentum", "ichimoku", "pattern", "rsi", "leadLag", "bayesian", "fusion", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.20, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "TREND_MOD_HIGH_VOL": {"baseWeightMultiplier": 0.85, "activeSignalTypes": ["trend", "ichimoku", "meanRev", "stochastic", "volPersist", "zScore", "ai", "pattern", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 0.85, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "RANGE_LOW_VOL": {"baseWeightMultiplier": 1.20, "activeSignalTypes": ["meanRev", "pattern", "volBreak", "stochastic", "harmonic", "fractalDim", "zScore", "bayesian", "fusion", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.20, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "RANGE_MED_VOL": {"baseWeightMultiplier": 1.10, "activeSignalTypes": ["meanRev", "pattern", "stochastic", "rsi", "bollinger", "harmonic", "zScore", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 1.10, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "RANGE_HIGH_VOL": {"baseWeightMultiplier": 0.75, "activeSignalTypes": ["meanRev", "entropy", "bollinger", "vwapDev", "volPersist", "zScore", "fusion", "ai", "pattern", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 0.75, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "WEAK_HIGH_VOL": {"baseWeightMultiplier": 0.70, "activeSignalTypes": ["meanRev", "entropy", "stochastic", "volPersist", "fractalDim", "zScore", "ai", "pattern", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 0.70, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "WEAK_MED_VOL": {"baseWeightMultiplier": 0.80, "activeSignalTypes": ["momentum", "meanRev", "pattern", "rsi", "fractalDim", "bayesian", "ai", "frequency", "balance", "oscillation", "microTrend", "deviation"], "contextualAggression": 0.80, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "WEAK_LOW_VOL": {"baseWeightMultiplier": 0.90, "activeSignalTypes": ["all"], "contextualAggression": 0.90, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0},
        "DEFAULT": {"baseWeightMultiplier": 1.0, "activeSignalTypes": ["all"], "contextualAggression": 1.0, "recentAccuracy": [], "totalPredictions": 0, "correctPredictions": 0}
    }
    driftDetector = { 'p_min': float('inf'), 's_min': float('inf'), 'n': 0, 'warning_level': 2.0, 'drift_level': 3.0 }
    consecutiveHighConfLosses = 0
    reflexiveCorrectionActive = 0
    engineMode = "NORMAL"
    qTable = {}
    GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE = 0.5
    heuristicPerformance = {}
    systemConsecutiveLosses = 0
    shared_stats_payload = {
        'lastPeriodFull': None,
        'lastPredictedOutcome': None,
        'lastActualOutcome': None,
        'lastFinalConfidence': None,
        'lastConfidenceLevel': None,
        'lastMacroRegime': None,
        'lastPredictionSignals': [],
        'lastConcentrationModeEngaged': False,
        'lastMarketEntropyState': None,
        'lastVolatilityRegime': None,
        'longTermGlobalAccuracy': 0.5,
        'consecutiveSamePredictions': 0,
        'consecutiveLosses': 0,
        'historicalMaxStreaks': {'BIG': 0, 'SMALL': 0},
        'risk_aversion_factor': 1.0,
        'consecutiveSkips': 0
    }
    
    # --- New Q-Learning & Engine State Initialization ---
    learningRate = 0.1
    discountFactor = 0.9
    explorationRate = 0.1
    MIN_DATA_FOR_COMPLEX_AI = 20
    ENGINE_PERFORMANCE = {}
    ENGINES = ['MasterAI', 'QuantumAI', 'NeuralNet', 'Fibonacci']
    for engine in ENGINES:
        ENGINE_PERFORMANCE[engine] = {'wins': 0, 'losses': 0, 'accuracy': 0.5, 'lossStreak': 0}

initialize_state()

def get_big_small_from_number(number):
    if number is None: return None
    if isinstance(number, str) and number.upper() in ['BIG', 'SMALL']:
        return number.upper()
    try:
        num = int(number)
    except (ValueError, TypeError):
        return None
    if 0 <= num <= 4: return 'SMALL'
    if 5 <= num <= 9: return 'BIG'
    return None

def get_opposite_outcome(prediction):
    if prediction == "BIG": return "SMALL"
    if prediction == "SMALL": return "BIG"
    return None

def determine_prediction_strategy(signals, final_decision):
    if not signals or not final_decision:
        return "Mixed"
    agreeing_signals = [s for s in signals if s.get('prediction') == final_decision]
    if not agreeing_signals:
        return "Corrective"
    categories = {'Trend Following': 0, 'Mean Reversion': 0, 'Pattern Matching': 0, 'Momentum': 0, 'Volatility': 0, 'Probabilistic': 0}
    source_map = {
        'Trend Following': ['MACD', 'Ichimoku', 'Fusion', 'StateSpace'],
        'Mean Reversion': ['Bollinger', 'MADev', 'ZScore', 'VWAPDev'],
        'Pattern Matching': ['Pattern', 'Streak', 'Gram', 'Cycle', 'Alt', 'Harmonic', 'WeightedHist', 'PhaseSpace', 'Heuristic', 'Fib-Retracement', 'ComplexPattern'],
        'Momentum': ['RSI', 'Stochastic'],
        'Volatility': ['Vol', 'Fractal', 'QuantumTunnel', 'Entropy'],
        'Probabilistic': ['Bayesian', 'MonteCarlo', 'Entangled', 'Superposition']
    }
    for signal in agreeing_signals:
        source = signal.get('source', '')
        for cat, keywords in source_map.items():
            if any(keyword in source for keyword in keywords):
                categories[cat] += signal.get('adjustedWeight', 0)
                break 
    if not any(v > 0 for v in categories.values()):
        return "Heuristic"
    dominant_strategy = max(categories, key=categories.get)
    return dominant_strategy


def calculate_sma(data, period):
    if not isinstance(data, list) or len(data) < period or period <= 0: return None
    relevant_data = data[:period]
    return sum(relevant_data) / period

def calculate_ema(data, period):
    if not isinstance(data, list) or len(data) < period or period <= 0: return None
    k = 2 / (period + 1)
    chronological_data = data[::-1]
    
    initial_slice_for_sma = chronological_data[:period][::-1]
    ema = calculate_sma(initial_slice_for_sma, period)
    if ema is None and len(initial_slice_for_sma) > 0:
        ema = sum(initial_slice_for_sma) / len(initial_slice_for_sma)
        
    if ema is None: return None
    for i in range(period, len(chronological_data)):
        ema = (chronological_data[i] * k) + (ema * (1 - k))
    return ema

def calculate_stddev(data, period):
    if not isinstance(data, list) or len(data) < period or period <= 0 or len(data) < 2: return None
    relevant_data = data[:period]
    mean = sum(relevant_data) / len(relevant_data)
    variance = sum([(x - mean) ** 2 for x in relevant_data]) / (len(relevant_data) - 1)
    return math.sqrt(variance)

def calculate_rsi(data, period):
    if period <= 0: return None
    chronological_data = data[::-1]
    if not isinstance(chronological_data, list) or len(chronological_data) < period + 1: return None
    gains, losses = 0, 0
    for i in range(1, period + 1):
        change = chronological_data[i] - chronological_data[i - 1]
        if change > 0: gains += change
        else: losses += abs(change)
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(chronological_data)):
        change = chronological_data[i] - chronological_data[i - 1]
        current_gain = change if change > 0 else 0
        current_loss = abs(change) if change < 0 else 0
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_current_ist_hour():
    try:
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        return {'raw': hour, 'sin': math.sin(hour / 24 * 2 * math.pi), 'cos': math.cos(hour / 24 * 2 * math.pi)}
    except Exception:
        hour = datetime.now().hour
        return { 'raw': hour, 'sin': math.sin(hour / 24 * 2 * math.pi), 'cos': math.cos(hour / 24 * 2 * math.pi) }

def get_real_time_external_data():
    return {'factor': 1.0, 'reason': 'Simulated'}

def get_prime_time_session(ist_hour):
    if 10 <= ist_hour < 12: return {"session": "PRIME_MORNING", "aggression": 1.25, "confidence": 1.15}
    if 13 <= ist_hour < 14: return {"session": "PRIME_AFTERNOON_1", "aggression": 1.15, "confidence": 1.1}
    if 15 <= ist_hour < 16: return {"session": "PRIME_AFTERNOON_2", "aggression": 1.15, "confidence": 1.1}
    if 17 <= ist_hour < 20:
        if ist_hour == 19: return {"session": "PRIME_EVENING_PEAK", "aggression": 1.35, "confidence": 1.25}
        return {"session": "PRIME_EVENING", "aggression": 1.3, "confidence": 1.2}
    return None

def analyze_trend_stability(history):
    if not isinstance(history, list) or len(history) < 25:
        return {"isStable": True, "reason": "Not enough data for stability check.", "details": "", "dominance": "NONE"}
    confirmed_history = [p for p in history if p and p.get('actual_outcome')]
    if len(confirmed_history) < 20:
        return {"isStable": True, "reason": "Not enough confirmed results.", "details": f"Confirmed: {len(confirmed_history)}", "dominance": "NONE"}
    recent_results = [get_big_small_from_number(p.get('actual_number')) for p in confirmed_history[:20] if get_big_small_from_number(p.get('actual_number'))]
    if len(recent_results) < 18:
        return {"isStable": True, "reason": "Not enough valid B/S for stability.", "details": f"Valid B/S: {len(recent_results)}", "dominance": "NONE"}
    big_count = recent_results.count("BIG")
    small_count = recent_results.count("SMALL")
    outcome_dominance = "NONE"
    if big_count / len(recent_results) >= 0.80:
        outcome_dominance = "BIG_DOMINANCE"
        return {"isStable": False, "reason": "Unstable: Extreme Outcome Dominance", "details": f"BIG:{big_count}, SMALL:{small_count} in last {len(recent_results)}", "dominance": outcome_dominance}
    if small_count / len(recent_results) >= 0.80:
        outcome_dominance = "SMALL_DOMINANCE"
        return {"isStable": False, "reason": "Unstable: Extreme Outcome Dominance", "details": f"BIG:{big_count}, SMALL:{small_count} in last {len(recent_results)}", "dominance": outcome_dominance}
    entropy = calculateEntropyForSignal(recent_results, len(recent_results))
    if entropy is not None and entropy < 0.45:
        return {"isStable": False, "reason": "Unstable: Very Low Entropy (Highly Predictable/Stuck)", "details": f"Entropy: {entropy:.2f}", "dominance": outcome_dominance}
    actual_numbers_recent = [int(p.get('actual_number')) for p in confirmed_history[:15] if 'actual_number' in p and isinstance(p.get('actual_number'), (int, float))]
    if len(actual_numbers_recent) >= 10:
        std_dev_num = calculate_stddev(actual_numbers_recent, len(actual_numbers_recent))
        if std_dev_num is not None and std_dev_num > 3.3:
            return {"isStable": False, "reason": "Unstable: High Numerical Volatility", "details": f"StdDev: {std_dev_num:.2f}", "dominance": outcome_dominance}
    alternations = 0
    for i in range(len(recent_results) - 1):
        if recent_results[i] != recent_results[i + 1]:
            alternations += 1
    if alternations / len(recent_results) > 0.75:
        return {"isStable": False, "reason": "Unstable: Excessive Choppiness", "details": f"Alternations: {alternations}/{len(recent_results)}", "dominance": outcome_dominance}
    return {"isStable": True, "reason": "Trend appears stable.", "details": f"Entropy: {entropy:.2f}" if entropy is not None else 'N/A', "dominance": outcome_dominance}

def analyze_market_entropy_state(history, trend_context, stability):
    ENTROPY_WINDOW_SHORT = 10
    ENTROPY_WINDOW_LONG = 25
    VOL_CHANGE_THRESHOLD = 0.3
    if len(history) < ENTROPY_WINDOW_LONG:
        return {"state": "UNCERTAIN_ENTROPY", "details": "Insufficient history for entropy state."}
    outcomes_short = [get_big_small_from_number(p.get('actual_number')) for p in history[:ENTROPY_WINDOW_SHORT] if get_big_small_from_number(p.get('actual_number'))]
    outcomes_long = [get_big_small_from_number(p.get('actual_number')) for p in history[:ENTROPY_WINDOW_LONG] if get_big_small_from_number(p.get('actual_number'))]
    entropy_short = calculateEntropyForSignal(outcomes_short, len(outcomes_short))
    entropy_long = calculateEntropyForSignal(outcomes_long, len(outcomes_long))
    numbers_short = [int(p.get('actual_number')) for p in history[:ENTROPY_WINDOW_SHORT] if 'actual_number' in p and isinstance(p.get('actual_number'), (int, float))]
    numbers_long_prev = [int(p.get('actual_number')) for p in history[ENTROPY_WINDOW_SHORT:ENTROPY_WINDOW_SHORT*2] if 'actual_number' in p and isinstance(p.get('actual_number'), (int, float))]
    short_term_volatility = None
    prev_short_term_volatility = None
    if len(numbers_short) >= ENTROPY_WINDOW_SHORT * 0.8:
        short_term_volatility = calculate_stddev(numbers_short, len(numbers_short))
    if len(numbers_long_prev) >= ENTROPY_WINDOW_SHORT * 0.8:
        prev_short_term_volatility = calculate_stddev(numbers_long_prev, len(numbers_long_prev))
    state = "STABLE_MODERATE"
    details = f"E_S:{entropy_short:.2f} E_L:{entropy_long:.2f} Vol_S:{short_term_volatility:.2f} Vol_P:{prev_short_term_volatility:.2f}" if entropy_short is not None and entropy_long is not None and short_term_volatility is not None and prev_short_term_volatility is not None else 'N/A'
    if entropy_short is None or entropy_long is None:
        return {"state": "UNCERTAIN_ENTROPY", "details": details}
    if entropy_short < 0.5 and entropy_long < 0.6 and short_term_volatility is not None and short_term_volatility < 1.5:
        state = "ORDERLY"
    elif entropy_short > 0.95 and entropy_long > 0.9:
        if short_term_volatility is not None and prev_short_term_volatility is not None and short_term_volatility > prev_short_term_volatility * (1 + VOL_CHANGE_THRESHOLD) and short_term_volatility > 2.5:
            state = "RISING_CHAOS"
        else:
            state = "STABLE_CHAOS"
    elif short_term_volatility is not None and prev_short_term_volatility is not None:
        if short_term_volatility > prev_short_term_volatility * (1 + VOL_CHANGE_THRESHOLD) and entropy_short > 0.85 and short_term_volatility > 2.0:
            state = "RISING_CHAOS"
        elif short_term_volatility < prev_short_term_volatility * (1 - VOL_CHANGE_THRESHOLD) and entropy_long > 0.85 and entropy_short < 0.80:
            state = "SUBSIDING_CHAOS"
    if not stability.get('isStable') and state in ["ORDERLY", "STABLE_MODERATE"]:
        state = "POTENTIAL_CHAOS_FROM_INSTABILITY"
        details += f" | StabilityOverride: {stability.get('reason')}"
    return {"state": state, "details": details}

def calculateEntropyForSignal(outcomes, windowSize):
    if not isinstance(outcomes, list) or len(outcomes) < windowSize:
        return None
    counts = {'BIG': 0, 'SMALL': 0}
    for outcome in outcomes[:windowSize]:
        if outcome:
            counts[outcome] = counts.get(outcome, 0) + 1
    entropy = 0
    total_valid_outcomes = counts['BIG'] + counts['SMALL']
    if total_valid_outcomes == 0:
        return 1
    for key in counts:
        if counts[key] > 0:
            p = counts[key] / total_valid_outcomes
            entropy -= p * math.log2(p)
    return entropy if not math.isnan(entropy) else 1

def analyze_transitions(history, base_weight):
    if not isinstance(history, list) or len(history) < 15: return None
    transitions = { "BIG": { "BIG": 0, "SMALL": 0, "total": 0 }, "SMALL": { "BIG": 0, "SMALL": 0, "total": 0 } }
    for i in range(len(history) - 1):
        currentBS = get_big_small_from_number(history[i].get('actual_number'))
        prevBS = get_big_small_from_number(history[i + 1].get('actual_number'))
        if currentBS and prevBS and transitions.get(prevBS):
            transitions[prevBS][currentBS] += 1
            transitions[prevBS]["total"] += 1
    lastOutcome = get_big_small_from_number(history[0].get('actual_number'))
    if not lastOutcome or not transitions.get(lastOutcome) or transitions[lastOutcome]["total"] < 6: return None
    nextBigProb = transitions[lastOutcome]["BIG"] / transitions[lastOutcome]["total"]
    nextSmallProb = transitions[lastOutcome]["SMALL"] / transitions[lastOutcome]["total"]
    if nextBigProb > nextSmallProb + 0.30: return {"prediction": "BIG", "weight": base_weight * nextBigProb, "source": "Transition"}
    if nextSmallProb > nextBigProb + 0.30: return {"prediction": "SMALL", "weight": base_weight * nextSmallProb, "source": "Transition"}
    return None

def analyze_streaks(history, base_weight):
    if not isinstance(history, list) or len(history) < 3: return None
    actuals = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(actuals) < 3: return None
    currentStreakType = actuals[0]
    currentStreakLength = 0
    for outcome in actuals:
        if outcome == currentStreakType:
            currentStreakLength += 1
        else:
            break
    if currentStreakLength >= 2:
        prediction = get_opposite_outcome(currentStreakType)
        weightFactor = min(0.45 + (currentStreakLength * 0.18), 0.95)
        return {"prediction": prediction, "weight": base_weight * weightFactor, "source": f"StreakBreak-{currentStreakLength}"}
    return None

def analyze_adaptive_streaks(history, shared_stats, base_weight):
    if not isinstance(history, list) or len(history) < 10: return None
    actuals = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(actuals) < 10: return None
    
    currentStreakType = actuals[0]
    currentStreakLength = 0
    for outcome in actuals:
        if outcome == currentStreakType:
            currentStreakLength += 1
        else:
            break
    
    historical_max = shared_stats.get('historicalMaxStreaks', {'BIG': 0, 'SMALL': 0})
    avg_streak = (historical_max.get('BIG', 0) + historical_max.get('SMALL', 0)) / 2 if (historical_max.get('BIG', 0) + historical_max.get('SMALL', 0)) > 0 else 0
    
    prediction = None
    confidence_factor = 0
    
    if currentStreakLength >= 3 and currentStreakLength > avg_streak:
        prediction = get_opposite_outcome(currentStreakType)
        confidence_factor = min(1.0, (currentStreakLength - avg_streak) * 0.2 + 0.5)
    
    elif currentStreakLength == 1:
        outcomes_10 = actuals[:10]
        big_count = outcomes_10.count("BIG")
        small_count = outcomes_10.count("SMALL")
        if abs(big_count - small_count) < 3: # Choppy market
            prediction = currentStreakType
            confidence_factor = 0.4
    
    if prediction:
        return {"prediction": prediction, "weight": base_weight * confidence_factor, "source": f"AdaptiveStreaks-{currentStreakLength}"}
    return None

def analyze_alternating_patterns(history, base_weight):
    if not isinstance(history, list) or len(history) < 5: return None
    actuals = [get_big_small_from_number(p.get('actual_number')) for p in history[:5] if p.get('actual_number')]
    if len(actuals) < 4: return None
    if actuals[0] == "SMALL" and actuals[1] == "BIG" and actuals[2] == "SMALL" and actuals[3] == "BIG":
        return {"prediction": "SMALL", "weight": base_weight * 1.15, "source": "Alt-BSBS->S"}
    if actuals[0] == "BIG" and actuals[1] == "SMALL" and actuals[2] == "BIG" and actuals[3] == "SMALL":
        return {"prediction": "BIG", "weight": base_weight * 1.15, "source": "Alt-SBSB->B"}
    return None

def analyze_heuristic_patterns(history, base_weight):
    if not isinstance(history, list) or len(history) < 5: return None
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history[:5] if p.get('actual_number')]
    if len(outcomes) < 5 or any(o is None for o in outcomes): return None
    
    if outcomes[0] != outcomes[1] and outcomes[1] == outcomes[2] and outcomes[2] != outcomes[3] and outcomes[3] == outcomes[4]:
        return {"prediction": outcomes[0], "weight": base_weight * 1.25, "source": "Heuristic-AABBA"}
    
    if outcomes[0] == outcomes[3] and outcomes[1] == outcomes[2] and outcomes[0] != outcomes[1]:
        return {"prediction": outcomes[0], "weight": base_weight * 1.15, "source": "Heuristic-ABBA"}
    
    if outcomes[0] != outcomes[1] and outcomes[1] != outcomes[2] and outcomes[2] != outcomes[3] and outcomes[3] != outcomes[4] and outcomes[0] == outcomes[2] and outcomes[2] == outcomes[4]:
        return {"prediction": outcomes[0], "weight": base_weight * 1.05, "source": "Heuristic-ABABA"}

    return None

# --- NEW: Added Complex Pattern Recognition ---
def analyze_complex_patterns(history, base_weight):
    if len(history) < 12:
        return None
    
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history[:12]]
    outcomes_str = "".join(['B' if o == "BIG" else 'S' for o in outcomes])

    pattern_4 = outcomes_str[:4]
    if pattern_4 == outcomes_str[4:8]:
        next_prediction_char = outcomes_str[8]
        prediction = "BIG" if next_prediction_char == 'B' else "SMALL"
        return {"prediction": prediction, "weight": base_weight * 1.4, "source": f"ComplexPattern-4p-{pattern_4}"}

    pattern_6 = outcomes_str[:6]
    if pattern_6 == outcomes_str[6:12]:
        next_prediction_char = pattern_6[0]
        prediction = "BIG" if next_prediction_char == 'B' else "SMALL"
        return {"prediction": prediction, "weight": base_weight * 1.6, "source": f"ComplexPattern-6p-{pattern_6}"}

    return None

# --- NEW: Added Fibonacci Signal ---
def analyze_fibonacci_patterns(history, base_weight):
    if len(history) < 20:
        return None
    
    numbers = [int(p.get('actual_number')) for p in history if p.get('actual_number') is not None]
    if len(numbers) < 20:
        return None
    
    recent_numbers = numbers[:10]
    high = max(recent_numbers)
    low = min(recent_numbers)
    trend_range = high - low
    
    if trend_range < 2:
        return None
        
    last_number = numbers[0]
    
    if numbers[9] < numbers[0]: # Uptrend
        retracement_levels = {
            '38.2%': high - (trend_range * 0.382),
            '61.8%': high - (trend_range * 0.618),
        }
        if last_number > retracement_levels['61.8%'] and last_number < retracement_levels['38.2%'] and last_number > numbers[1]:
            return {"prediction": "BIG", "weight": base_weight * 1.5, "source": "Fib-Retracement-Uptrend"}

    if numbers[9] > numbers[0]: # Downtrend
        retracement_levels = {
            '38.2%': low + (trend_range * 0.382),
            '61.8%': low + (trend_range * 0.618),
        }
        if last_number < retracement_levels['61.8%'] and last_number > retracement_levels['38.2%'] and last_number < numbers[1]:
            return {"prediction": "SMALL", "weight": base_weight * 1.5, "source": "Fib-Retracement-Downtrend"}
    
    return None

def analyze_weighted_historical(history, weightDecayFactor, base_weight):
    if not isinstance(history, list) or len(history) < 5: return None
    bigWeightedScore = 0
    smallWeightedScore = 0
    currentWeight = 1.0
    maxHistory = min(len(history), 20)
    for i in range(maxHistory):
        outcome = get_big_small_from_number(history[i].get('actual_number'))
        if outcome == "BIG": bigWeightedScore += currentWeight
        elif outcome == "SMALL": smallWeightedScore += currentWeight
        currentWeight *= weightDecayFactor
    if bigWeightedScore == 0 and smallWeightedScore == 0: return None
    totalScore = bigWeightedScore + smallWeightedScore + 0.0001
    if bigWeightedScore > smallWeightedScore: return {"prediction": "BIG", "weight": base_weight * (bigWeightedScore / totalScore), "source": "WeightedHist"}
    if smallWeightedScore > bigWeightedScore: return {"prediction": "SMALL", "weight": base_weight * (smallWeightedScore / totalScore), "source": "WeightedHist"}
    return None

def analyze_two_plus_one_patterns(history, base_weight):
    if not history or len(history) < 3: return None
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history[:3] if p.get('actual_number')]
    if len(outcomes) < 3 or any(o is None for o in outcomes): return None
    if outcomes[0] == 'BIG' and outcomes[1] == 'BIG' and outcomes[2] == 'SMALL': return {'prediction': 'BIG', 'weight': base_weight * 0.85, 'source': 'Pattern-BBS->B'}
    if outcomes[0] == 'SMALL' and outcomes[1] == 'SMALL' and outcomes[2] == 'BIG': return {'prediction': 'SMALL', 'weight': base_weight * 0.85, 'source': 'Pattern-SSB->S'}
    return None

def analyze_double_patterns(history, base_weight):
    if not history or len(history) < 4: return None
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history[:4] if p.get('actual_number')]
    if len(outcomes) < 4 or any(o is None for o in outcomes): return None
    if outcomes[0] == 'BIG' and outcomes[1] == 'BIG' and outcomes[2] == 'SMALL' and outcomes[3] == 'SMALL': return {'prediction': 'BIG', 'weight': base_weight * 1.1, 'source': 'Pattern-SSBB->B'}
    if outcomes[0] == 'SMALL' and outcomes[1] == 'SMALL' and outcomes[2] == 'BIG' and outcomes[3] == 'BIG': return {'prediction': 'SMALL', 'weight': base_weight * 1.1, 'source': 'Pattern-BBSS->S'}
    return None

def analyze_mirror_patterns(history, base_weight):
    if not history or len(history) < 4: return None
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history[:4] if p.get('actual_number')]
    if len(outcomes) < 4 or any(o is None for o in outcomes): return None
    if outcomes[0] == outcomes[3] and outcomes[1] == outcomes[2] and outcomes[0] != outcomes[1]:
        return {'prediction': outcomes[0], 'weight': base_weight * 1.2, 'source': f'Pattern-Mirror->{outcomes[0]}'}
    return None

def analyze_rsi(history, rsiPeriod, base_weight, volatility):
    if rsiPeriod <= 0: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(actual_numbers) < rsiPeriod + 1: return None
    rsiValue = calculate_rsi(actual_numbers, rsiPeriod)
    if rsiValue is None: return None
    overbought = 70
    oversold = 30
    if volatility == "HIGH": overbought, oversold = 80, 20
    elif volatility == "MEDIUM": overbought, oversold = 75, 25
    elif volatility == "LOW": overbought, oversold = 68, 32
    elif volatility == "VERY_LOW": overbought, oversold = 65, 35
    prediction = None
    signalStrengthFactor = 0
    if rsiValue < oversold:
        prediction = "BIG"
        signalStrengthFactor = (oversold - rsiValue) / oversold
    elif rsiValue > overbought:
        prediction = "SMALL"
        signalStrengthFactor = (rsiValue - overbought) / (100 - overbought)
    if prediction:
        return {"prediction": prediction, "weight": base_weight * (0.60 + min(signalStrengthFactor, 1.0) * 0.40), "source": "RSI"}
    return None

def analyze_macd(history, shortPeriod, longPeriod, signalPeriod, base_weight):
    if shortPeriod <= 0 or longPeriod <= 0 or signalPeriod <= 0 or shortPeriod >= longPeriod: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(actual_numbers) < longPeriod + signalPeriod - 1: return None
    emaShort = calculate_ema(actual_numbers, shortPeriod)
    emaLong = calculate_ema(actual_numbers, longPeriod)
    if emaShort is None or emaLong is None: return None
    macdLineCurrent = emaShort - emaLong
    macdLineValues = []
    for i in range(longPeriod - 1, len(actual_numbers)):
        currentSlice = actual_numbers[len(actual_numbers) - 1 - i:]
        shortE = calculate_ema(currentSlice, shortPeriod)
        longE = calculate_ema(currentSlice, longPeriod)
        if shortE is not None and longE is not None:
            macdLineValues.append(shortE - longE)
    if len(macdLineValues) < signalPeriod: return None
    signalLine = calculate_ema(macdLineValues[::-1], signalPeriod)
    if signalLine is None: return None
    macdHistogram = macdLineCurrent - signalLine
    prediction = None
    if len(macdLineValues) >= signalPeriod + 1:
        prevMacdSliceForSignal = macdLineValues[:-1]
        prevSignalLine = calculate_ema(prevMacdSliceForSignal[::-1], signalPeriod)
        prevMacdLine = macdLineValues[-2]
        if prevSignalLine is not None and prevMacdLine is not None:
            if prevMacdLine <= prevSignalLine and macdLineCurrent > signalLine: prediction = "BIG"
            elif prevMacdLine >= prevSignalLine and macdLineCurrent < signalLine: prediction = "SMALL"
    if not prediction:
        if macdHistogram > 0.25: prediction = "BIG"
        elif macdHistogram < -0.25: prediction = "SMALL"
    if prediction:
        strengthFactor = min(abs(macdHistogram) / 0.6, 1.0)
        return {"prediction": prediction, "weight": base_weight * (0.55 + strengthFactor * 0.45), "source": f"MACD_{'CrossB' if prediction == 'BIG' else 'CrossS'}", "macdHistogram": macdHistogram}
    return None

def analyze_bollinger_bands(history, period, stdDevMultiplier, base_weight):
    if period <= 0: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(actual_numbers) < period: return None
    sma = calculate_sma(actual_numbers[:period], period)
    if sma is None: return None
    stdDev = calculate_stddev(actual_numbers, period)
    if stdDev is None or stdDev < 0.05: return None
    upperBand = sma + (stdDev * stdDevMultiplier)
    lowerBand = sma - (stdDev * stdDevMultiplier)
    lastNumber = actual_numbers[0]
    prediction = None
    if lastNumber > upperBand * 1.01: prediction = "SMALL"
    elif lastNumber < lowerBand * 0.99: prediction = "BIG"
    if prediction:
        bandBreachStrength = abs(lastNumber - sma) / (stdDev * stdDevMultiplier + 0.001)
        return {"prediction": prediction, "weight": base_weight * (0.65 + min(bandBreachStrength, 0.9)*0.35), "source": "Bollinger"}
    return None

def analyze_ichimoku_cloud(history, tenkanPeriod, kijunPeriod, senkouBPeriod, base_weight):
    if tenkanPeriod <= 0 or kijunPeriod <= 0 or senkouBPeriod <= 0: return None
    chronological_history = history[::-1]
    numbers = [int(entry.get('actual_number')) for entry in chronological_history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < max(senkouBPeriod, kijunPeriod) + kijunPeriod - 1: return None
    def get_high_low(data_slice):
        if not data_slice: return {'high': None, 'low': None}
        return {'high': max(data_slice), 'low': min(data_slice)}
    tenkanSenValues = []
    for i in range(len(numbers)):
        if i < tenkanPeriod - 1: tenkanSenValues.append(None); continue
        hl = get_high_low(numbers[i - tenkanPeriod + 1 : i + 1])
        if hl['high'] is not None and hl['low'] is not None: tenkanSenValues.append((hl['high'] + hl['low']) / 2)
        else: tenkanSenValues.append(None)
    kijunSenValues = []
    for i in range(len(numbers)):
        if i < kijunPeriod - 1: kijunSenValues.append(None); continue
        hl = get_high_low(numbers[i - kijunPeriod + 1 : i + 1])
        if hl['high'] is not None and hl['low'] is not None: kijunSenValues.append((hl['high'] + hl['low']) / 2)
        else: kijunSenValues.append(None)
    currentTenkan = tenkanSenValues[-1] if tenkanSenValues else None
    prevTenkan = tenkanSenValues[-2] if len(tenkanSenValues) > 1 else None
    currentKijun = kijunSenValues[-1] if kijunSenValues else None
    prevKijun = kijunSenValues[-2] if len(kijunSenValues) > 1 else None
    senkouSpanAValues = []
    for i in range(len(numbers)):
        if tenkanSenValues[i] is not None and kijunSenValues[i] is not None: senkouSpanAValues.append((tenkanSenValues[i] + kijunSenValues[i]) / 2)
        else: senkouSpanAValues.append(None)
    senkouSpanBValues = []
    for i in range(len(numbers)):
        if i < senkouBPeriod - 1: senkouSpanBValues.append(None); continue
        hl = get_high_low(numbers[i - senkouBPeriod + 1 : i + 1])
        if hl['high'] is not None and hl['low'] is not None: senkouSpanBValues.append((hl['high'] + hl['low']) / 2)
        else: senkouSpanBValues.append(None)
    currentSenkouA = senkouSpanAValues[len(numbers) - 1 - kijunPeriod] if len(numbers) > kijunPeriod and len(senkouSpanAValues) > len(numbers) - 1 - kijunPeriod else None
    currentSenkouB = senkouSpanBValues[len(numbers) - 1 - kijunPeriod] if len(numbers) > kijunPeriod and len(senkouSpanBValues) > len(numbers) - 1 - kijunPeriod else None
    chikouSpan = numbers[-1] if numbers else None
    priceKijunPeriodsAgo = numbers[len(numbers) - 1 - kijunPeriod] if len(numbers) > kijunPeriod else None
    lastPrice = numbers[-1] if numbers else None
    if lastPrice is None or currentTenkan is None or currentKijun is None or currentSenkouA is None or currentSenkouB is None or chikouSpan is None or priceKijunPeriodsAgo is None:
        return None
    prediction = None
    strengthFactor = 0.3
    tkCrossSignal = None
    if prevTenkan is not None and prevKijun is not None:
        if prevTenkan <= prevKijun and currentTenkan > currentKijun: tkCrossSignal = "BIG"
        elif prevTenkan >= prevKijun and currentTenkan < currentKijun: tkCrossSignal = "SMALL"
    cloudTop = max(currentSenkouA, currentSenkouB)
    cloudBottom = min(currentSenkouA, currentSenkouB)
    priceVsCloudSignal = None
    if lastPrice > cloudTop * 1.01: priceVsCloudSignal = "BIG"
    elif lastPrice < cloudBottom * 0.99: priceVsCloudSignal = "SMALL"
    chikouSignal = None
    if chikouSpan > priceKijunPeriodsAgo: chikouSignal = "BIG"
    elif chikouSpan < priceKijunPeriodsAgo: chikouSignal = "SMALL"
    if tkCrossSignal and tkCrossSignal == priceVsCloudSignal and tkCrossSignal == chikouSignal: prediction, strengthFactor = tkCrossSignal, 0.95
    elif priceVsCloudSignal and priceVsCloudSignal == tkCrossSignal and chikouSignal == priceVsCloudSignal: prediction, strengthFactor = priceVsCloudSignal, 0.85
    elif priceVsCloudSignal and priceVsCloudSignal == tkCrossSignal: prediction, strengthFactor = priceVsCloudSignal, 0.7
    elif priceVsCloudSignal and priceVsCloudSignal == chikouSignal: prediction, strengthFactor = priceVsCloudSignal, 0.65
    elif tkCrossSignal and priceVsCloudSignal: prediction, strengthFactor = tkCrossSignal, 0.55
    elif priceVsCloudSignal: prediction, strengthFactor = priceVsCloudSignal, 0.5
    if prediction == "BIG" and lastPrice > currentKijun and prevKijun is not None and numbers[-2] <= prevKijun and priceVsCloudSignal == "BIG":
        strengthFactor = min(1.0, strengthFactor + 0.15)
    elif prediction == "SMALL" and lastPrice < currentKijun and prevKijun is not None and numbers[-2] >= prevKijun and priceVsCloudSignal == "SMALL":
        strengthFactor = min(1.0, strengthFactor + 0.15)
    if prediction: return {"prediction": prediction, "weight": base_weight * strengthFactor, "source": "Ichimoku"}
    return None

def analyze_stochastic(history, kPeriod, dPeriod, smoothK, base_weight, volatility):
    if kPeriod <= 0 or dPeriod <= 0 or smoothK <= 0: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(actual_numbers) < kPeriod + smoothK - 1 + dPeriod - 1: return None
    chronological_numbers = actual_numbers[::-1]
    kValues = []
    for i in range(kPeriod - 1, len(chronological_numbers)):
        currentSlice = chronological_numbers[i - kPeriod + 1: i + 1]
        currentClose = currentSlice[-1]
        lowestLow = min(currentSlice)
        highestHigh = max(currentSlice)
        if highestHigh == lowestLow: kValues.append(kValues[-1] if kValues else 50)
        else: kValues.append(100 * (currentClose - lowestLow) / (highestHigh - lowestLow))
    if len(kValues) < smoothK: return None
    smoothedKValues = []
    for i in range(len(kValues) - smoothK + 1):
        smoothedKValues.append(calculate_sma(kValues[i:i + smoothK][::-1], smoothK))
    if len(smoothedKValues) < dPeriod: return None
    dValues = []
    for i in range(len(smoothedKValues) - dPeriod + 1):
        dValues.append(calculate_sma(smoothedKValues[i:i + dPeriod][::-1], dPeriod))
    if len(smoothedKValues) < 2 or len(dValues) < 2: return None
    currentK = smoothedKValues[-1]
    prevK = smoothedKValues[-2]
    currentD = dValues[-1]
    prevD = dValues[-2]
    overbought, oversold = 80, 20
    if volatility == "HIGH": overbought, oversold = 88, 12
    elif volatility == "MEDIUM": overbought, oversold = 82, 18
    elif volatility == "LOW": overbought, oversold = 75, 25
    elif volatility == "VERY_LOW": overbought, oversold = 70, 30
    prediction, strengthFactor = None, 0
    if prevK <= prevD and currentK > currentD and currentK < overbought - 5:
        prediction, strengthFactor = "BIG", max(0.35, (oversold + 5 - min(currentK, currentD, oversold + 5)) / (oversold + 5))
    elif prevK >= prevD and currentK < currentD and currentK > oversold + 5:
        prediction, strengthFactor = "SMALL", max(0.35, (max(currentK, currentD, overbought - 5) - (overbought - 5)) / (100 - (overbought - 5)))
    if not prediction:
        if prevK < oversold and currentK >= oversold and currentK < (oversold + (overbought - oversold) / 2):
            prediction, strengthFactor = "BIG", max(0.25, (currentK - oversold) / ((overbought - oversold) / 2))
        elif prevK > overbought and currentK <= overbought and currentK > (oversold + (overbought - oversold) / 2):
            prediction, strengthFactor = "SMALL", max(0.25, (overbought - currentK) / ((overbought - oversold) / 2))
    if prediction: return {"prediction": prediction, "weight": base_weight * (0.5 + min(strengthFactor, 1.0) * 0.5), "source": "Stochastic", "currentK": currentK}
    return None

def analyze_ma_deviation(history, longMAPeriod, normalizationPeriod, base_weight):
    if longMAPeriod <= 0 or normalizationPeriod <= 0: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(actual_numbers) < max(longMAPeriod, normalizationPeriod): return None
    lastNumber = actual_numbers[0]
    longMA = calculate_ema(actual_numbers, longMAPeriod)
    stdDevNorm = calculate_stddev(actual_numbers, normalizationPeriod)
    if longMA is None or stdDevNorm is None or stdDevNorm < 0.01: return None
    deviationScore = (lastNumber - longMA) / stdDevNorm
    prediction, strengthFactor = None, 0
    threshold = 1.8
    if deviationScore > threshold: prediction, strengthFactor = "SMALL", min((deviationScore - threshold) / threshold, 1.0)
    elif deviationScore < -threshold: prediction, strengthFactor = "BIG", min(abs(deviationScore - (-threshold)) / threshold, 1.0)
    if prediction: return {"prediction": prediction, "weight": base_weight * (0.4 + strengthFactor * 0.6), "source": "MADev"}
    return None

def analyze_vwap_deviation(history, vwapPeriod, normalizationPeriod, base_weight):
    def calculate_vwap(data, period):
        if not data or len(data) < period: return None
        total_price_volume = 0
        total_volume = 0
        for i in range(period):
            entry = data[i]
            price = float(entry.get('actual_number'))
            volume = float(entry.get('volume', 1))
            if not math.isnan(price) and not math.isnan(volume) and volume > 0:
                total_price_volume += price * volume
                total_volume += volume
        if total_volume == 0: return None
        return total_price_volume / total_volume

    if vwapPeriod <= 0 or normalizationPeriod <= 0: return None
    actual_numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(history) < max(vwapPeriod, normalizationPeriod) or len(actual_numbers) < 1: return None
    vwap = calculate_vwap(history, vwapPeriod)
    stdDevPrice = calculate_stddev(actual_numbers, normalizationPeriod)
    if vwap is None or stdDevPrice is None or stdDevPrice < 0.01: return None
    lastNumber = actual_numbers[0]
    deviationScore = (lastNumber - vwap) / stdDevPrice
    prediction, strengthFactor = None, 0
    threshold = 1.5
    if deviationScore > threshold: prediction, strengthFactor = "SMALL", min((deviationScore - threshold) / threshold, 1.0)
    elif deviationScore < -threshold: prediction, strengthFactor = "BIG", min(abs(deviationScore - (-threshold)) / threshold, 1.0)
    if prediction: return {"prediction": prediction, "weight": base_weight * (0.45 + strengthFactor * 0.55), "source": "VWAPDev"}
    return None

def analyze_harmonic_potential(history, base_weight):
    numbers = [float(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < 20: return None
    swings = []
    chronological_numbers = numbers[::-1]
    for i in range(2, len(chronological_numbers) - 2):
        is_peak = chronological_numbers[i] > chronological_numbers[i-1] and chronological_numbers[i] > chronological_numbers[i-2] and \
                  chronological_numbers[i] > chronological_numbers[i+1] and chronological_numbers[i] > chronological_numbers[i+2]
        is_trough = chronological_numbers[i] < chronological_numbers[i-1] and chronological_numbers[i] < chronological_numbers[i-2] and \
                    chronological_numbers[i] < chronological_numbers[i+1] and chronological_numbers[i] < chronological_numbers[i+2]
        if is_peak or is_trough:
            new_swing = {'price': chronological_numbers[i], 'index': len(numbers) - 1 - i, 'type': 'peak' if is_peak else 'trough'}
            if not swings or (is_peak and swings[0]['type'] == 'trough') or (is_trough and swings[0]['type'] == 'peak'):
                swings.insert(0, new_swing)
            else:
                if is_peak and swings[0]['type'] == 'peak' and new_swing['price'] > swings[0]['price']: swings[0] = new_swing
                if is_trough and swings[0]['type'] == 'trough' and new_swing['price'] < swings[0]['price']: swings[0] = new_swing
    if len(swings) < 3: return None
    C, B, X = swings[0], swings[1], swings[2]
    if not X or not B or not C or X['type'] == B['type'] or B['type'] == C['type']: return None
    XA_val = abs(B['price'] - X['price'])
    AB_val = XA_val
    BC_val = abs(C['price'] - B['price'])
    if AB_val < 0.8 or BC_val < 0.5: return None
    lastPrice = numbers[0]
    prediction, strengthFactor = None, 0
    bcRetracementOfAb = BC_val / AB_val
    if X['type'] == 'peak' and B['type'] == 'trough' and C['type'] == 'peak':
        if bcRetracementOfAb >= 0.382 and bcRetracementOfAb <= 0.886:
            prz_D_gartley = X['price'] - AB_val * 0.786
            if lastPrice >= prz_D_gartley * 0.98 and lastPrice <= prz_D_gartley * 1.02 and lastPrice < B['price']:
                prediction, strengthFactor = "BIG", 0.6
    elif X['type'] == 'trough' and B['type'] == 'peak' and C['type'] == 'trough':
        if bcRetracementOfAb >= 0.382 and bcRetracementOfAb <= 0.886:
            prz_D_gartley = X['price'] + AB_val * 0.786
            if lastPrice <= prz_D_gartley * 1.02 and lastPrice >= prz_D_gartley * 0.98 and lastPrice > B['price']:
                prediction, strengthFactor = "SMALL", 0.6
    if prediction: return {"prediction": prediction, "weight": base_weight * max(0.25, min(strengthFactor, 0.85)), "source": "HarmonicPotV3"}
    return None

def analyze_ngram_patterns(history, n, base_weight):
    if not isinstance(history, list) or len(history) < n + 10: return None
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(outcomes) < n + 5: return None
    recentNGram = "-".join(outcomes[:n])
    patternCounts = {}
    for i in range(len(outcomes) - (n + 1) + 1):
        pattern = "-".join(outcomes[i+1 : i+1+n])
        nextOutcome = outcomes[i]
        if pattern not in patternCounts: patternCounts[pattern] = {'BIG': 0, 'SMALL': 0, 'total': 0}
        patternCounts[pattern][nextOutcome] += 1
        patternCounts[pattern]['total'] += 1
    if recentNGram in patternCounts and patternCounts[recentNGram]['total'] >= 4:
        data = patternCounts[recentNGram]
        probBig = data['BIG'] / data['total']
        probSmall = data['SMALL'] / data['total']
        if probBig > probSmall + 0.30 and probBig > 0.65: return {'prediction': 'BIG', 'weight': base_weight * probBig * 1.1, 'source': f'{n}GramB'}
        if probSmall > probBig + 0.30 and probSmall > 0.65: return {'prediction': 'SMALL', 'weight': base_weight * probSmall * 1.1, 'source': f'{n}GramS'}
    return None

def analyze_cyclical_patterns(history, period, base_weight):
    if len(history) < period or period < 8: return None
    outcomes = [get_big_small_from_number(p.get('actual_outcome')) for p in history[:period] if p.get('actual_outcome')]
    if len(outcomes) < period * 0.80: return None
    for cycleLen in range(3, 7):
        if len(outcomes) < cycleLen * 2.8: continue
        cycle1String = "".join(outcomes[:cycleLen])
        cycle2String = "".join(outcomes[cycleLen:cycleLen*2])
        if len(cycle1String) == cycleLen and cycle1String == cycle2String:
            matchLength = 0
            for k in range(cycleLen):
                if (cycleLen * 2 + k) < len(outcomes) and outcomes[cycleLen * 2 + k] == outcomes[k]: matchLength += 1
                else: break
            if matchLength >= math.floor(cycleLen * 0.66):
                predictedOutcome = outcomes[cycleLen - 1]
                if predictedOutcome: return {'prediction': predictedOutcome, 'weight': base_weight * (0.65 + (1 / cycleLen) + (matchLength / cycleLen * 0.2)), 'source': f'Cycle{cycleLen}StrongCont'}
    return None

def analyze_volatility_persistence(history, period, base_weight):
    numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < period * 2: return None
    recentVolSlice = numbers[:period]
    prevVolSlice = numbers[period:period * 2]
    currentStdDev = calculate_stddev(recentVolSlice, period)
    prevStdDev = calculate_stddev(prevVolSlice, period)
    if currentStdDev is None or prevStdDev is None: return None
    prediction, strengthFactor = None, 0
    if currentStdDev > prevStdDev * 1.3 and currentStdDev > 2.0:
        if numbers[0] > numbers[1]: prediction = "BIG"
        elif numbers[0] < numbers[1]: prediction = "SMALL"
        strengthFactor = 0.3
    elif currentStdDev < prevStdDev * 0.7 and currentStdDev < 1.0:
        if numbers[0] > numbers[1]: prediction = "SMALL"
        elif numbers[0] < numbers[1]: prediction = "BIG"
        strengthFactor = 0.35
    if prediction: return {"prediction": prediction, "weight": base_weight * strengthFactor, "source": "VolPersist"}
    return None

def analyze_fractal_dimension(history, period, base_weight):
    numbers = [float(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < period + 1: return None
    chronological_numbers = numbers[::-1]
    periodSlice = chronological_numbers[-period:]
    highestHighP = max(periodSlice) if periodSlice else -float('inf')
    lowestLowP = min(periodSlice) if periodSlice else float('inf')
    if highestHighP == -float('inf') or lowestLowP == float('inf'): return None
    N1_val = (highestHighP - lowestLowP) / period
    if N1_val == 0: return {"value": 1.0, "interpretation": "EXTREMELY_TRENDING_OR_FLAT", "prediction": None, "weight": 0, "source": "FractalDim"}
    sumPriceChanges = sum(abs(periodSlice[i] - periodSlice[i-1]) for i in range(1, len(periodSlice)))
    N2_val = sumPriceChanges / period
    if N2_val == 0 and N1_val != 0: return {"value": 2.0, "interpretation": "CHOPPY_MAX_NOISE", "prediction": None, "weight": 0, "source": "FractalDim"}
    if N2_val == 0 and N1_val == 0: return {"value": 1.0, "interpretation": "FLAT_NO_MOVEMENT", "prediction": None, "weight": 0, "source": "FractalDim"}
    priceChangeOverPeriod = abs(periodSlice[-1] - periodSlice[0])
    ER = priceChangeOverPeriod / sumPriceChanges if sumPriceChanges > 0 else 0
    FDI_approx = 1 + (1 - ER)
    interpretation = "UNKNOWN"
    prediction = None
    if FDI_approx < 1.35: interpretation = "TRENDING"
    elif FDI_approx > 1.65: interpretation = "CHOPPY_RANGING"
    else: interpretation = "MODERATE_ACTIVITY"
    if FDI_approx > 1.75:
        lastOutcome = get_big_small_from_number(numbers[0])
        if lastOutcome: prediction = get_opposite_outcome(lastOutcome)
    return {"value": FDI_approx, "interpretation": interpretation, "prediction": prediction, "weight": base_weight * 0.2 if prediction else 0, "source": "FractalDim"}

def analyze_entropy_signal(history, period, base_weight):
    if len(history) < period: return None
    outcomes = [get_big_small_from_number(e.get('actual_number')) for e in history[:period] if e.get('actual_number')]
    entropy = calculateEntropyForSignal(outcomes, period)
    if entropy is None: return None
    if entropy < 0.55:
        lastOutcome = outcomes[0]
        if lastOutcome: return {"prediction": get_opposite_outcome(lastOutcome), "weight": base_weight * (1 - entropy) * 0.85, "source": "EntropyReversal"}
    elif entropy > 0.98:
        lastOutcome = outcomes[0]
        if lastOutcome: return {"prediction": lastOutcome, "weight": base_weight * 0.25, "source": "EntropyHighContWeak"}
    return None

def analyze_volatility_breakout(history, trendContext, base_weight):
    if trendContext.get('volatility') == "VERY_LOW" and len(history) >= 3:
        lastOutcome = get_big_small_from_number(history[0].get('actual_number'))
        prevOutcome = get_big_small_from_number(history[1].get('actual_number'))
        if lastOutcome and prevOutcome and lastOutcome == prevOutcome: return {"prediction": lastOutcome, "weight": base_weight * 0.8, "source": "VolSqueezeBreakoutCont"}
        if lastOutcome and prevOutcome and lastOutcome != prevOutcome: return {"prediction": lastOutcome, "weight": base_weight * 0.6, "source": "VolSqueezeBreakoutInitial"}
    return None

def analyze_waveform_patterns(history, base_weight):
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(outcomes) < 8: return None
    wave = [1 if o == "BIG" else -1 for o in outcomes[:8]]
    if wave[0] == wave[1] and wave[2] == wave[3] and wave[0] != wave[2]:
        return {"prediction": "BIG" if wave[0] == 1 else "SMALL", "weight": base_weight * 1.2, "source": "Waveform-Constructive"}
    if wave[0] != wave[1] and wave[1] != wave[2] and wave[2] != wave[3]:
        return {"prediction": "SMALL" if wave[0] == 1 else "BIG", "weight": base_weight, "source": "Waveform-Destructive"}
    return None

def analyze_phase_space(history, base_weight):
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(outcomes) < 10: return None
    recent = outcomes[:10]
    bigCount = recent.count("BIG")
    smallCount = recent.count("SMALL")
    if bigCount >= 7: return {"prediction": "BIG", "weight": base_weight * ((bigCount - 5) / 5), "source": "PhaseSpace-BigAttractor"}
    if smallCount >= 7: return {"prediction": "SMALL", "weight": base_weight * ((smallCount - 5) / 5), "source": "PhaseSpace-SmallAttractor"}
    return None

def analyze_quantum_tunneling(history, base_weight):
    actuals = [p.get('actual_number') for p in history if p.get('actual_number') is not None]
    if len(actuals) < 2: return None
    lastNum = actuals[0]
    prevNum = actuals[1]
    if (lastNum <= 1 and prevNum >= 8) or (lastNum >= 8 and prevNum <= 1):
        return {"prediction": "SMALL" if lastNum > 4 else "BIG", "weight": base_weight, "source": "QuantumTunneling"}
    return None

def analyze_entanglement(history, lag, base_weight):
    outcomes = [get_big_small_from_number(p.get('actual_number')) for p in history if p.get('actual_number')]
    if len(outcomes) < lag + 10: return None
    match, antiMatch = 0, 0
    for i in range(10):
        if outcomes[i] == outcomes[i + lag]: match += 1
        else: antiMatch += 1
    if match >= 8: return {"prediction": outcomes[lag - 1], "weight": base_weight, "source": f"Entangled-Corr-Lag{lag}"}
    if antiMatch >= 8: return {"prediction": "SMALL" if outcomes[lag - 1] == "BIG" else "BIG", "weight": base_weight, "source": f"Entangled-AntiCorr-Lag{lag}"}
    return None

def analyze_monte_carlo_signal(signals, base_weight):
    if len(signals) < 5: return None
    bigProb = sum(s['adjustedWeight'] for s in signals if s['prediction'] == "BIG")
    smallProb = sum(s['adjustedWeight'] for s in signals if s['prediction'] == "SMALL")
    totalWeight = bigProb + smallProb
    if totalWeight == 0: return None
    normalizedBigProb = bigProb / totalWeight
    bigWins = 0
    simulations = 1000
    for _ in range(simulations):
        if random.random() < normalizedBigProb: bigWins += 1
    if bigWins / simulations > 0.7: return {"prediction": "BIG", "weight": base_weight * (bigWins / simulations), "source": "MonteCarlo"}
    if bigWins / simulations < 0.3: return {"prediction": "SMALL", "weight": base_weight * (1 - (bigWins / simulations)), "source": "MonteCarlo"}
    return None

def analyze_volatility_trend_fusion(trendContext, marketEntropyState, base_weight):
    direction, strength, volatility = trendContext.get('direction'), trendContext.get('strength'), trendContext.get('volatility')
    entropy = marketEntropyState.get('state')
    prediction, weightFactor = None, 0
    if strength == 'STRONG' and (volatility == 'LOW' or volatility == 'MEDIUM') and entropy == 'ORDERLY':
        prediction = 'BIG' if 'BIG' in direction else 'SMALL'
        weightFactor = 1.4
    elif strength == 'STRONG' and volatility == 'HIGH' and 'CHAOS' in entropy:
        prediction = 'SMALL' if 'BIG' in direction else 'BIG'
        weightFactor = 1.2
    elif strength == 'RANGING' and volatility == 'LOW' and entropy == 'ORDERLY':
        prediction = 'BIG' if random.random() > 0.5 else 'SMALL'
        weightFactor = 0.8
    if prediction: return {"prediction": prediction, "weight": base_weight * weightFactor, "source": 'Vol-Trend-Fusion'}
    return None

def analyze_ml_model_signal(features, base_weight):
    if not features: return None
    rsi_14, macd_hist, stddev_30, time_sin = features.get('rsi_14'), features.get('macd_hist'), features.get('stddev_30'), features.get('time_sin')
    if rsi_14 is None or macd_hist is None or stddev_30 is None or time_sin is None: return None
    modelConfidence, prediction = 0, None
    if rsi_14 > 70 and macd_hist < -0.1:
        prediction = "SMALL"
        modelConfidence = abs(macd_hist) + (rsi_14 - 70) / 30
    elif rsi_14 < 30 and macd_hist > 0.1:
        prediction = "BIG"
        modelConfidence = abs(macd_hist) + (30 - rsi_14) / 30
    elif stddev_30 < 1.0 and time_sin > 0:
        prediction = "BIG"
        modelConfidence = 0.4
    if prediction:
        weight = base_weight * min(1.0, modelConfidence) * 1.5
        return {"prediction": prediction, "weight": weight, "source": "ML-GradientBoost"}
    return None

def analyze_bayesian_inference(signals, base_weight):
    if not signals or len(signals) < 5: return None
    posteriorBig, posteriorSmall = 0.5, 0.5
    categories = {'trend': {'BIG': 0, 'SMALL': 0, 'total': 0}, 'momentum': {'BIG': 0, 'SMALL': 0, 'total': 0}, 'meanRev': {'BIG': 0, 'SMALL': 0, 'total': 0}}
    def get_category(source):
        if "MACD" in source or "Ichimoku" in source: return 'trend'
        if "Stochastic" in source or "RSI" in source: return 'momentum'
        if "Bollinger" in source or "MADev" in source or "ZScore" in source: return 'meanRev'
        return None
    for s in signals:
        category = get_category(s['source'])
        if category and s['prediction'] in ['BIG', 'SMALL']:
            categories[category][s['prediction']] += s['adjustedWeight']
            categories[category]['total'] += s['adjustedWeight']
    for cat in categories.values():
        if cat['total'] > 0:
            evidenceForBig = cat['BIG'] / cat['total']
            evidenceForSmall = cat['SMALL'] / cat['total']
            newPosteriorBig = evidenceForBig * posteriorBig
            newPosteriorSmall = evidenceForSmall * posteriorSmall
            normalization = newPosteriorBig + newPosteriorSmall
            if normalization > 0:
                posteriorBig = newPosteriorBig / normalization
                posteriorSmall = newPosteriorSmall / normalization
    if posteriorBig > posteriorSmall and posteriorBig > 0.65: return {"prediction": "BIG", "weight": base_weight * (posteriorBig - 0.5) * 2, "source": "Bayesian"}
    if posteriorSmall > posteriorBig and posteriorSmall > 0.65: return {"prediction": "SMALL", "weight": base_weight * (posteriorSmall - 0.5) * 2, "source": "Bayesian"}
    return None

def analyze_zscore_anomaly(history, period, threshold, base_weight):
    numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < period: return None
    slice = numbers[:period]
    mean = calculate_sma(slice, period)
    stdDev = calculate_stddev(slice, period)
    if mean is None or stdDev is None or stdDev < 0.1: return None
    lastNumber = numbers[0]
    zScore = (lastNumber - mean) / stdDev
    prediction, strengthFactor = None, 0
    threshold = 1.8
    if zScore > threshold: prediction, strengthFactor = "SMALL", min((zScore - threshold) / threshold, 1.0)
    elif zScore < -threshold: prediction, strengthFactor = "BIG", min(abs(zScore - (-threshold)) / threshold, 1.0)
    if prediction: return {"prediction": prediction, "weight": base_weight * (0.5 + strengthFactor * 0.5), "source": "ZScoreAnomaly"}
    return None

def analyze_state_space_momentum(history, period, base_weight):
    numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < period * 2: return None
    chronological_numbers = numbers[::-1]
    velocity, error, gain = 0, 0, 0.6
    for i in range(1, len(chronological_numbers)):
        measurement = chronological_numbers[i] - chronological_numbers[i - 1]
        prediction = velocity
        error = measurement - prediction
        velocity = prediction + gain * error
    velocities = [chronological_numbers[i] - chronological_numbers[i - 1] for i in range(1, len(chronological_numbers))]
    avgVelocity = sum(velocities) / len(velocities)
    prediction = None
    if velocity > avgVelocity * 1.8 and velocity > 0.5: prediction = "BIG"
    elif velocity < avgVelocity * 1.8 and velocity < -0.5: prediction = "SMALL"
    if prediction:
        strengthFactor = min(1, abs(velocity - avgVelocity) / (abs(avgVelocity) + 1))
        return {"prediction": prediction, "weight": base_weight * strengthFactor, "source": "StateSpaceMomentum"}
    return None

def analyze_prediction_consensus(signals, trendContext):
    if not signals or len(signals) < 4: return {"score": 0.5, "factor": 1.0, "details": "Insufficient signals for consensus"}
    categories = {
        'trend': {'BIG': 0, 'SMALL': 0, 'weight': 0}, 'momentum': {'BIG': 0, 'SMALL': 0, 'weight': 0},
        'meanRev': {'BIG': 0, 'SMALL': 0, 'weight': 0}, 'pattern': {'BIG': 0, 'SMALL': 0, 'weight': 0},
        'volatility': {'BIG': 0, 'SMALL': 0, 'weight': 0}, 'probabilistic': {'BIG': 0, 'SMALL': 0, 'weight': 0},
        'ml': {'BIG': 0, 'SMALL': 0, 'weight': 0}
    }
    def get_category(source):
        if "MACD" in source or "Ichimoku" in source or "StateSpace" in source or "Fusion" in source: return 'trend'
        if "Stochastic" in source or "RSI" in source: return 'momentum'
        if "Bollinger" in source or "MADev" in source or "ZScore" in source or "VWAPDev" in source: return 'meanRev'
        if "Gram" in source or "Cycle" in source or "Alt" in source or "Harmonic" in source or "Pattern" in source or "Streak" in source or "Transition" in source or "WeightedHist" in source: return 'pattern'
        if "Vol" in source or "Fractal" in source or "QuantumTunnel" in source or "Entropy" in source: return 'volatility'
        if "Bayesian" in source or "Superposition" in source or "ML-" in source or "MonteCarlo" in source or "Entangled" in source: return 'probabilistic'
        if "ML-" in source: return 'ml'
        return None
    for s in signals:
        category = get_category(s['source'])
        if category and s['prediction'] in ["BIG", "SMALL"]:
            categories[category][s['prediction']] += s['adjustedWeight']
    bigCats, smallCats, mixedCats = 0, 0, 0
    for cat in categories.values():
        totalWeight = cat['BIG'] + cat['SMALL']
        if totalWeight > 0:
            if cat['BIG'] > cat['SMALL'] * 1.2: bigCats += 1
            elif cat['SMALL'] > cat['BIG'] * 1.2: smallCats += 1
            else: mixedCats += 1
    consensusScore = 0.5
    totalCats = bigCats + smallCats + mixedCats
    if totalCats > 0:
        dominantCats, nonDominantCats = max(bigCats, smallCats), min(bigCats, smallCats)
        consensusScore = (dominantCats - nonDominantCats) / totalCats
    factor = 1.0 + (consensusScore * 0.4)
    if trendContext.get('strength') == 'STRONG':
        if (categories['trend']['BIG'] > categories['trend']['SMALL'] and categories['momentum']['SMALL'] > categories['momentum']['BIG']) or \
           (categories['trend']['SMALL'] > categories['trend']['BIG'] and categories['momentum']['BIG'] > categories['momentum']['SMALL']):
            factor *= 0.6
    return {"score": consensusScore, "factor": max(0.4, min(1.6, factor)), "details": f"Bcat:{bigCats},Scat:{smallCats},Mcat:{mixedCats},Score:{consensusScore:.2f}"}

def analyze_predictive_divergence(signals, final_decision):
    if not signals or not final_decision:
        return {"divergence_score": 0.0, "reason": "No signals"}
    
    trend_weight_agree = 0
    trend_weight_disagree = 0
    mean_rev_weight_agree = 0
    mean_rev_weight_disagree = 0
    
    def get_category(source):
        if "MACD" in source or "Ichimoku" in source or "StateSpace" in source or "Fusion" in source: return 'trend'
        if "Bollinger" in source or "MADev" in source or "ZScore" in source or "VWAPDev" in source or "RSI" in source or "Stochastic" in source: return 'mean_rev'
        return None
        
    for s in signals:
        category = get_category(s['source'])
        if category == 'trend':
            if s['prediction'] == final_decision:
                trend_weight_agree += s['adjustedWeight']
            else:
                trend_weight_disagree += s['adjustedWeight']
        elif category == 'mean_rev':
            if s['prediction'] == final_decision:
                mean_rev_weight_agree += s['adjustedWeight']
            else:
                mean_rev_weight_disagree += s['adjustedWeight']
                
    divergence_score = 0.0
    reason = "No significant divergence"
    
    if trend_weight_agree > 0 and mean_rev_weight_disagree > 0:
        divergence_score = min(1.0, (trend_weight_agree + mean_rev_weight_disagree) / (trend_weight_agree + trend_weight_disagree + mean_rev_weight_agree + mean_rev_weight_disagree))
        if divergence_score > 0.4:
            reason = "High Trend vs Mean-Reversion divergence"
    
    return {"divergence_score": divergence_score, "reason": reason}

def analyze_quantum_superposition_state(signals, consensus, base_weight):
    if not signals or len(signals) < 5 or not consensus: return None
    totalWeight = sum(s.get('adjustedWeight', 0) for s in signals)
    if totalWeight < 0.1: return None
    bigWeight = sum(s.get('adjustedWeight', 0) for s in signals if s.get('prediction') == "BIG")
    smallWeight = sum(s.get('adjustedWeight', 0) for s in signals if s.get('prediction') == "SMALL")
    bigCollapseProbability = (bigWeight / totalWeight) * consensus.get('factor')
    smallCollapseProbability = (smallWeight / totalWeight) * (2.0 - consensus.get('factor'))
    if bigCollapseProbability > smallCollapseProbability * 1.3:
        return {"prediction": "BIG", "weight": base_weight * min(1.0, (bigCollapseProbability - smallCollapseProbability)), "source": "QuantumSuperposition"}
    if smallCollapseProbability > bigCollapseProbability * 1.3:
        return {"prediction": "SMALL", "weight": base_weight * min(1.0, (smallCollapseProbability - bigCollapseProbability)), "source": "QuantumSuperposition"}
    return None

def analyze_path_confluence_strength(signals, finalPrediction):
    if not signals or len(signals) == 0 or not finalPrediction: return {"score": 0, "diversePaths": 0, "details": "No valid signals or prediction."}
    agreeingSignals = [s for s in signals if s.get('prediction') == finalPrediction and s.get('adjustedWeight', 0) > MIN_ABSOLUTE_WEIGHT * 10]
    if len(agreeingSignals) < 2: return {"score": 0, "diversePaths": len(agreeingSignals), "details": "Insufficient agreeing signals."}
    signalCategories = set()
    for s in agreeingSignals:
        source = s.get('source')
        if "MACD" in source or "Ichimoku" in source: signalCategories.add('trend')
        elif "Stochastic" in source or "RSI" in source: signalCategories.add('momentum')
        elif "Bollinger" in source or "ZScore" in source or "MADev" in source or "VWAPDev" in source: signalCategories.add('meanRev')
        elif "Gram" in source or "Cycle" in source or "Alt" in source or "Harmonic" in source or "Pattern" in source or "Streak" in source or "Transition" in source or "WeightedHist" in source: signalCategories.add('pattern')
        elif "Vol" in source or "FractalDim" in source or "QuantumTunnel" in source or "Entropy" in source: signalCategories.add('volatility')
        elif "Bayesian" in source or "Superposition" in source or "ML-" in source or "MonteCarlo" in source or "Entangled" in source: signalCategories.add('probabilistic')
        else: signalCategories.add('other')
    diversePathCount = len(signalCategories)
    confluenceScore = 0
    if diversePathCount >= 4: confluenceScore = 0.20
    elif diversePathCount == 3: confluenceScore = 0.12
    elif diversePathCount == 2: confluenceScore = 0.05
    veryStrongAgreeingCount = len([s for s in agreeingSignals if s.get('adjustedWeight', 0) > 0.10])
    confluenceScore += min(veryStrongAgreeingCount * 0.02, 0.10)
    return {"score": min(confluenceScore, 0.30), "diversePaths": diversePathCount, "details": f"Paths:{diversePathCount},Strong:{veryStrongAgreeingCount}"}

def analyze_signal_consistency(signals, trendContext):
    if not signals or len(signals) < 3: return {"score": 0.70, "details": "Too few signals for consistency check"}
    validSignals = [s for s in signals if s.get('prediction')]
    if len(validSignals) < 3: return {"score": 0.70, "details": "Too few valid signals"}
    predictions = {'BIG': 0, 'SMALL': 0}
    for s in validSignals:
        if s.get('prediction') in ['BIG', 'SMALL']: predictions[s['prediction']] += 1
    totalPredictions = predictions['BIG'] + predictions['SMALL']
    if totalPredictions == 0: return {"score": 0.5, "details": "No directional signals"}
    consistencyScore = max(predictions['BIG'], predictions['SMALL']) / totalPredictions
    return {"score": consistencyScore, "details": f"Overall split B:{predictions['BIG']}/S:{predictions['SMALL']}"}

def checkForAnomalousPerformance(currentSharedStats):
    global reflexiveCorrectionActive, consecutiveHighConfLosses
    if reflexiveCorrectionActive > 0:
        reflexiveCorrectionActive -= 1
        return True
    if currentSharedStats and isinstance(currentSharedStats.get('lastFinalConfidence'), (int, float)) and currentSharedStats.get('lastActualOutcome'):
        lastPredOutcomeBS = get_big_small_from_number(currentSharedStats.get('lastActualOutcome'))
        lastPredWasCorrect = lastPredOutcomeBS == currentSharedStats.get('lastPredictedOutcome')
        lastPredWasHighConf = currentSharedStats.get('lastConfidenceLevel') == 3
        if lastPredWasHighConf and not lastPredWasCorrect: consecutiveHighConfLosses += 1
        else: consecutiveHighConfLosses = 0
    if consecutiveHighConfLosses >= 2:
        reflexiveCorrectionActive = 5
        consecutiveHighConfLosses = 0
        return True
    return False

# --- NEW: Upgraded Loop Protection and Correction ---
def analyze_and_correct_prediction_loops(shared_stats, signals):
    consecutive_losses = shared_stats.get('consecutiveLosses', 0)
    consecutive_same_predictions = shared_stats.get('consecutiveSamePredictions', 0)
    last_predicted_outcome = shared_stats.get('lastPredictedOutcome')
    last_contributing_signals = shared_stats.get('lastPredictionSignals', [])

    if consecutive_losses >= 2 and consecutive_same_predictions >= 2 and last_predicted_outcome and last_contributing_signals:
        print(f"Loop detected: {consecutive_losses} losses with the same prediction '{last_predicted_outcome}'. Penalizing signals.")
        losing_signal_sources = { s['source'] for s in last_contributing_signals if s.get('prediction') == last_predicted_outcome }
        for signal in signals:
            if signal['source'] in losing_signal_sources:
                penalty_factor = 0.25
                original_weight = signal.get('adjustedWeight', 0)
                signal['adjustedWeight'] = original_weight * penalty_factor
    return signals

# --- NEW: Prediction Diversity Logic ---
def apply_prediction_diversity_logic(final_confidence, shared_stats):
    consecutive_same = shared_stats.get('consecutiveSamePredictions', 0)
    if consecutive_same > 3:
        penalty = 1.0 - (min(consecutive_same, 8) * 0.02)
        return final_confidence * penalty, f"DiversityPenalty({penalty:.2f})"
    return final_confidence, None

def calculate_uncertainty_score(trendContext, stability, marketEntropyState, signalConsistency, pathConfluence, globalAccuracy, isReflexiveCorrection, driftState, predictiveDivergence):
    uncertaintyScore, reasons = 0, []
    if isReflexiveCorrection: uncertaintyScore += 80; reasons.append("ReflexiveCorrection")
    if driftState == 'DRIFT': uncertaintyScore += 70; reasons.append("ConceptDrift")
    elif driftState == 'WARNING': uncertaintyScore += 40; reasons.append("DriftWarning")
    if not stability.get('isStable'):
        uncertaintyScore += 50 if ("Dominance" in stability.get('reason') or "Choppiness" in stability.get('reason')) else 40
        reasons.append(f"Instability:{stability.get('reason')}")
    if "CHAOS" in marketEntropyState.get('state', ''):
        uncertaintyScore += 45 if marketEntropyState.get('state') == "RISING_CHAOS" else 35
        reasons.append(marketEntropyState.get('state'))
    if signalConsistency.get('score', 0) < 0.6:
        uncertaintyScore += (1 - signalConsistency.get('score')) * 50
        reasons.append(f"LowConsistency:{signalConsistency.get('score', 0):.2f}")
    if pathConfluence.get('diversePaths', 0) < 3:
        uncertaintyScore += (3 - pathConfluence.get('diversePaths')) * 15
        reasons.append(f"LowConfluence:{pathConfluence.get('diversePaths')}")
    if trendContext.get('isTransitioning'): uncertaintyScore += 25; reasons.append("RegimeTransition")
    if trendContext.get('volatility') == "HIGH": uncertaintyScore += 20; reasons.append("HighVolatility")
    if isinstance(globalAccuracy, (int, float)) and globalAccuracy < 0.48:
        uncertaintyScore += (0.48 - globalAccuracy) * 150
        reasons.append(f"LowGlobalAcc:{globalAccuracy:.2f}")
    if predictiveDivergence.get('divergence_score', 0) > 0.4:
        uncertaintyScore += predictiveDivergence['divergence_score'] * 40
        reasons.append(f"HighDivergence:{predictiveDivergence['divergence_score']:.2f}")
    return {"score": uncertaintyScore, "reasons": ";".join(reasons)}

def create_feature_set_for_ml(history, trendContext, time):
    numbers = [int(e.get('actual_number')) for e in history if e.get('actual_number') is not None]
    if len(numbers) < 52: return None
    rsi_14 = calculate_rsi(numbers, 14)
    macd_result = analyze_macd(history, 12, 26, 9, 1.0)
    stddev_10 = calculate_stddev(numbers, 10)
    stddev_30 = calculate_stddev(numbers, 30)
    if any(x is None for x in [rsi_14, macd_result, stddev_10, stddev_30]): return None
    return {
        'time_sin': time['sin'], 'time_cos': time['cos'],
        'last_5_mean': calculate_sma(numbers, 5), 'last_20_mean': calculate_sma(numbers, 20),
        'stddev_10': stddev_10, 'stddev_30': stddev_30,
        'rsi_14': rsi_14,
        'stoch_k_14': analyze_stochastic(history, 14, 3, 3, 1.0, trendContext.get('volatility')).get('currentK') if analyze_stochastic(history, 14, 3, 3, 1.0, trendContext.get('volatility')) else None,
        'macd_hist': macd_result.get('macdHistogram') if macd_result else None,
        'trend_strength': 2 if trendContext.get('strength') == 'STRONG' else (1 if trendContext.get('strength') == 'MODERATE' else 0),
        'volatility_level': 2 if trendContext.get('volatility') == 'HIGH' else (1 if trendContext.get('volatility') == 'MEDIUM' else 0),
    }

def perform_six_level_bypass_check(history, stats, stability, entropy, driftState):
    MIN_HISTORY = 52
    checks = {'passed': True, 'reasons': []}
    if not history or len(history) < MIN_HISTORY:
        checks['passed'] = False
        checks['reasons'].append(f'BypassFail-L1:InsufficientHistory({len(history)}/{MIN_HISTORY})')
    if not stability.get('isStable'):
        checks['passed'] = False
        checks['reasons'].append(f'BypassFail-L2:MarketUnstable({stability.get("reason")})')
    if "CHAOS" in entropy.get('state', ''):
        checks['passed'] = False
        checks['reasons'].append(f'BypassFail-L3:MarketChaotic({entropy.get("state")})')
    if driftState == 'DRIFT':
        checks['passed'] = False
        checks['reasons'].append('BypassFail-L4:ConceptDriftDetected')
    if checkForAnomalousPerformance(stats):
        checks['passed'] = False
        checks['reasons'].append('BypassFail-L5:ReflexiveCorrectionActive')
    globalAccuracy = stats.get('longTermGlobalAccuracy', 0.5)
    if globalAccuracy < 0.45:
        checks['passed'] = False
        checks['reasons'].append(f'BypassFail-L6:LowGlobalAccuracy({globalAccuracy:.2f})')
    return checks

def get_dynamic_weight_adjustment(signalSourceName, baseWeight, currentPeriodFull, currentVolatilityRegime, sessionHistory):
    global signalPerformance, MIN_ABSOLUTE_WEIGHT, PROBATION_THRESHOLD_ACCURACY, PROBATION_MIN_OBSERVATIONS, PROBATION_WEIGHT_CAP, MIN_WEIGHT_FACTOR, MAX_WEIGHT_FACTOR, ALPHA_UPDATE_RATE, MAX_ALPHA_FACTOR, MIN_ALPHA_FACTOR
    
    perf = signalPerformance.get(signalSourceName)
    if not perf:
        signalPerformance[signalSourceName] = {
            'correct': 0, 'total': 0, 'recentAccuracy': [],
            'sessionCorrect': 0, 'sessionTotal': 0,
            'lastUpdatePeriod': None, 'lastActivePeriod': None,
            'currentAdjustmentFactor': 1.0, 'alphaFactor': 1.0, 'longTermImportanceScore': 0.5,
            'performanceByVolatility': {}, 'isOnProbation': False
        }
        return max(baseWeight, MIN_ABSOLUTE_WEIGHT)

    if sessionHistory and len(sessionHistory) <= 1:
        perf['sessionCorrect'] = 0
        perf['sessionTotal'] = 0

    if perf['lastUpdatePeriod'] != currentPeriodFull:
        perf['lastUpdatePeriod'] = currentPeriodFull

    volatilitySpecificAdjustment = 1.0
    if perf['performanceByVolatility'].get(currentVolatilityRegime) and perf['performanceByVolatility'][currentVolatilityRegime]['total'] >= MIN_OBSERVATIONS_FOR_ADJUST / 2.0:
        volPerf = perf['performanceByVolatility'][currentVolatilityRegime]
        volAccuracy = volPerf['correct'] / volPerf['total']
        volDeviation = volAccuracy - 0.5
        volatilitySpecificAdjustment = 1 + (volDeviation * 1.30)
        volatilitySpecificAdjustment = min(max(volatilitySpecificAdjustment, 0.55), 1.45)

    sessionAdjustmentFactor = 1.0
    if perf['sessionTotal'] >= 3:
        sessionAccuracy = perf['sessionCorrect'] / perf['sessionTotal']
        sessionDeviation = sessionAccuracy - 0.5
        sessionAdjustmentFactor = 1 + (sessionDeviation * 1.5)
        sessionAdjustmentFactor = min(max(sessionAdjustmentFactor, 0.6), 1.4)

    finalAdjustmentFactor = perf['currentAdjustmentFactor'] * perf['alphaFactor'] * volatilitySpecificAdjustment * sessionAdjustmentFactor * (0.70 + perf['longTermImportanceScore'] * 0.6)

    if perf['isOnProbation']: finalAdjustmentFactor = min(finalAdjustmentFactor, PROBATION_WEIGHT_CAP)

    adjustedWeight = baseWeight * finalAdjustmentFactor
    return max(adjustedWeight, MIN_ABSOLUTE_WEIGHT)

def update_signal_performance(contributingSignals, actualOutcome, periodFull, currentVolatilityRegime, lastFinalConfidence, concentrationModeActive, marketEntropyState):
    global signalPerformance, PERFORMANCE_WINDOW, MIN_OBSERVATIONS_FOR_ADJUST, MIN_WEIGHT_FACTOR, MAX_WEIGHT_FACTOR, PROBATION_MIN_OBSERVATIONS, PROBATION_THRESHOLD_ACCURACY, ALPHA_UPDATE_RATE, MAX_ALPHA_FACTOR, MIN_ALPHA_FACTOR
    
    if not actualOutcome or not contributingSignals: return
    isHighConfidencePrediction = lastFinalConfidence > 0.75
    
    isOverallCorrect = actualOutcome == shared_stats_payload.get('lastPredictedOutcome')

    for signal in contributingSignals:
        if not signal or not signal.get('source'): continue
        source = signal['source']
        if source not in signalPerformance:
            signalPerformance[source] = {
                'correct': 0, 'total': 0, 'recentAccuracy': [],
                'sessionCorrect': 0, 'sessionTotal': 0,
                'lastUpdatePeriod': None, 'lastActivePeriod': None,
                'currentAdjustmentFactor': 1.0, 'alphaFactor': 1.0, 'longTermImportanceScore': 0.5,
                'performanceByVolatility': {}, 'isOnProbation': False
            }
        
        if currentVolatilityRegime not in signalPerformance[source]['performanceByVolatility']:
            signalPerformance[source]['performanceByVolatility'][currentVolatilityRegime] = {'correct': 0, 'total': 0}

        if signalPerformance[source]['lastUpdatePeriod'] != periodFull:
            signalPerformance[source]['lastUpdatePeriod'] = periodFull

        if signalPerformance[source]['lastActivePeriod'] != periodFull:
            signalPerformance[source]['total'] += 1
            signalPerformance[source]['sessionTotal'] += 1
            signalPerformance[source]['performanceByVolatility'][currentVolatilityRegime]['total'] += 1
            outcomeCorrect = 1 if signal.get('prediction') == actualOutcome else 0
            if outcomeCorrect:
                signalPerformance[source]['correct'] += 1
                signalPerformance[source]['sessionCorrect'] += 1
                signalPerformance[source]['performanceByVolatility'][currentVolatilityRegime]['correct'] += 1
            
            importanceDelta = 0.025 if outcomeCorrect and isHighConfidencePrediction else (0.01 if outcomeCorrect else (-0.040 if isHighConfidencePrediction and not isOverallCorrect else -0.015))
            if concentrationModeActive or "CHAOS" in marketEntropyState.get('state', ''): importanceDelta *= 1.5
            
            signalPerformance[source]['longTermImportanceScore'] = min(1.0, max(0.0, signalPerformance[source]['longTermImportanceScore'] + importanceDelta))
            signalPerformance[source]['recentAccuracy'].append(outcomeCorrect)
            if len(signalPerformance[source]['recentAccuracy']) > PERFORMANCE_WINDOW: signalPerformance[source]['recentAccuracy'].pop(0)

            if signalPerformance[source]['total'] >= MIN_OBSERVATIONS_FOR_ADJUST and len(signalPerformance[source]['recentAccuracy']) >= PERFORMANCE_WINDOW / 2:
                recentCorrectCount = sum(signalPerformance[source]['recentAccuracy'])
                accuracy = recentCorrectCount / len(signalPerformance[source]['recentAccuracy'])
                deviation = accuracy - 0.5
                newAdjustmentFactor = 1 + (deviation * 3.5)
                newAdjustmentFactor = min(max(newAdjustmentFactor, MIN_WEIGHT_FACTOR), MAX_WEIGHT_FACTOR)
                signalPerformance[source]['currentAdjustmentFactor'] = newAdjustmentFactor
                
                if len(signalPerformance[source]['recentAccuracy']) >= PROBATION_MIN_OBSERVATIONS and accuracy < PROBATION_THRESHOLD_ACCURACY:
                    signalPerformance[source]['isOnProbation'] = True
                elif accuracy > PROBATION_THRESHOLD_ACCURACY + 0.15:
                    signalPerformance[source]['isOnProbation'] = False
                
                alphaLearningRate = ALPHA_UPDATE_RATE * (1.75 if accuracy < 0.35 else (1.4 if accuracy < 0.45 else 1))
                if newAdjustmentFactor > signalPerformance[source]['alphaFactor']:
                    signalPerformance[source]['alphaFactor'] = min(MAX_ALPHA_FACTOR, signalPerformance[source]['alphaFactor'] + alphaLearningRate * (newAdjustmentFactor - signalPerformance[source]['alphaFactor']))
                else:
                    signalPerformance[source]['alphaFactor'] = max(MIN_ALPHA_FACTOR, signalPerformance[source]['alphaFactor'] - alphaLearningRate * (signalPerformance[source]['alphaFactor'] - newAdjustmentFactor))
            signalPerformance[source]['lastActivePeriod'] = periodFull

def update_regime_profile_performance(regime, actualOutcome, predictedOutcome):
    global REGIME_SIGNAL_PROFILES, REGIME_ACCURACY_WINDOW, REGIME_LEARNING_RATE_BASE, GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE
    if regime in REGIME_SIGNAL_PROFILES and predictedOutcome:
        profile = REGIME_SIGNAL_PROFILES[regime]
        profile['totalPredictions'] = profile.get('totalPredictions', 0) + 1
        outcomeCorrect = 1 if actualOutcome == predictedOutcome else 0
        if outcomeCorrect: profile['correctPredictions'] = profile.get('correctPredictions', 0) + 1
        profile['recentAccuracy'].append(outcomeCorrect)
        if len(profile['recentAccuracy']) > REGIME_ACCURACY_WINDOW: profile['recentAccuracy'].pop(0)
        if len(profile['recentAccuracy']) >= REGIME_ACCURACY_WINDOW * 0.7:
            regimeAcc = sum(profile['recentAccuracy']) / len(profile['recentAccuracy'])
            dynamicLearningRateFactor = 1.0 + abs(0.5 - GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE) * 0.7
            dynamicLearningRateFactor = max(0.65, min(1.5, dynamicLearningRateFactor))
            currentLearningRate = REGIME_LEARNING_RATE_BASE * dynamicLearningRateFactor
            currentLearningRate = max(0.01, min(0.07, currentLearningRate))
            if regimeAcc > 0.62:
                profile['baseWeightMultiplier'] = min(1.9, profile['baseWeightMultiplier'] + currentLearningRate)
                profile['contextualAggression'] = min(1.8, profile['contextualAggression'] + currentLearningRate * 0.5)
            elif regimeAcc < 0.38:
                profile['baseWeightMultiplier'] = max(0.20, profile['baseWeightMultiplier'] - currentLearningRate * 1.3)
                profile['contextualAggression'] = max(0.30, profile['contextualAggression'] - currentLearningRate * 0.7)

def detectConceptDrift(isCorrect):
    global driftDetector
    driftDetector['n'] += 1
    errorRate = 0 if isCorrect else 1
    p_i = (driftDetector.get('p_i', 0) * (driftDetector['n'] - 1) + errorRate) / driftDetector['n']
    driftDetector['p_i'] = p_i
    s_i = math.sqrt(p_i * (1 - p_i) / driftDetector['n'])
    if p_i + s_i < driftDetector['p_min'] + driftDetector['s_min']:
        driftDetector['p_min'] = p_i
        driftDetector['s_min'] = s_i
    if p_i + s_i > driftDetector['p_min'] + driftDetector['drift_level'] * driftDetector['s_min']:
        driftDetector['p_min'] = float('inf')
        driftDetector['s_min'] = float('inf')
        driftDetector['n'] = 1
        return 'DRIFT'
    elif p_i + s_i > driftDetector['p_min'] + driftDetector['warning_level'] * driftDetector['s_min']:
        return 'WARNING'
    else:
        return 'STABLE'

def get_trend_context(history, shortMALookback=5, mediumMALookback=10, longMALookback=20):
    if not isinstance(history, list) or len(history) < longMALookback:
        return {"strength": "UNKNOWN", "direction": "NONE", "volatility": "UNKNOWN", "details": "Insufficient history", "macroRegime": "UNKNOWN_REGIME", "isTransitioning": False}
    
    numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) < longMALookback:
        return {"strength": "UNKNOWN", "direction": "NONE", "volatility": "UNKNOWN", "details": "Insufficient numbers", "macroRegime": "UNKNOWN_REGIME", "isTransitioning": False}

    shortMA = calculate_ema(numbers, shortMALookback)
    mediumMA = calculate_ema(numbers, mediumMALookback)
    longMA = calculate_ema(numbers, longMALookback)

    if shortMA is None or mediumMA is None or longMA is None:
        return {"strength": "UNKNOWN", "direction": "NONE", "volatility": "UNKNOWN", "details": "MA calculation failed", "macroRegime": "UNKNOWN_REGIME", "isTransitioning": False}

    direction = "NONE"
    strength = "WEAK"
    details = f"S:{shortMA:.1f},M:{mediumMA:.1f},L:{longMA:.1f}"

    stdDevLong = calculate_stddev(numbers, longMALookback)
    epsilon = 0.001
    normalizedSpread = (shortMA - longMA) / (stdDevLong or epsilon)

    details += f",NormSpread:{normalizedSpread:.2f}"

    if shortMA > mediumMA and mediumMA > longMA:
        direction = "BIG"
        if normalizedSpread > 0.80:
            strength = "STRONG"
        elif normalizedSpread > 0.45:
            strength = "MODERATE"
        else:
            strength = "WEAK"
    elif shortMA < mediumMA and mediumMA < longMA:
        direction = "SMALL"
        if normalizedSpread < -0.80:
            strength = "STRONG"
        elif normalizedSpread < -0.45:
            strength = "MODERATE"
        else:
            strength = "WEAK"
    else:
        strength = "RANGING"
        if shortMA > longMA:
            direction = "BIG_BIASED_RANGE"
        elif longMA > shortMA:
            direction = "SMALL_BIASED_RANGE"

    volatility = "UNKNOWN"
    volSlice = numbers[:min(len(numbers), 30)]
    if len(volSlice) >= 15:
        stdDevVol = calculate_stddev(volSlice, len(volSlice))
        if stdDevVol is not None:
            details += f" VolStdDev:{stdDevVol:.2f}"
            if stdDevVol > 3.3:
                volatility = "HIGH"
            elif stdDevVol > 2.0:
                volatility = "MEDIUM"
            else:
                volatility = "LOW" if stdDevVol > 0.9 else "VERY_LOW"
    
    return {"strength": strength, "direction": direction, "volatility": volatility, "details": details, "macroRegime": "PENDING_REGIME_CLASSIFICATION", "isTransitioning": False}

def get_market_regime_and_trend_context(history, shortMALookback=5, mediumMALookback=10, longMALookback=20):
    baseContext = get_trend_context(history, shortMALookback, mediumMALookback, longMALookback)
    macroRegime = "UNCERTAIN"
    strength = baseContext.get('strength')
    volatility = baseContext.get('volatility')
    isTransitioning = False
    numbers = [int(entry.get('actual_number')) for entry in history if 'actual_number' in entry and isinstance(entry.get('actual_number'), (int, float))]
    if len(numbers) > mediumMALookback + 5:
        prevShortMA = calculate_ema(numbers[1:], shortMALookback)
        prevMediumMA = calculate_ema(numbers[1:], mediumMALookback)
        currentShortMA = calculate_ema(numbers, shortMALookback)
        currentMediumMA = calculate_ema(numbers, mediumMALookback)
        if prevShortMA is not None and prevMediumMA is not None and currentShortMA is not None and currentMediumMA is not None:
            if (prevShortMA <= prevMediumMA and currentShortMA > currentMediumMA) or \
               (prevShortMA >= prevMediumMA and currentShortMA < currentMediumMA):
                isTransitioning = True
    if strength == "STRONG":
        if volatility in ["LOW", "VERY_LOW"]: macroRegime = "TREND_STRONG_LOW_VOL"
        elif volatility == "MEDIUM": macroRegime = "TREND_STRONG_MED_VOL"
        else: macroRegime = "TREND_STRONG_HIGH_VOL"
    elif strength == "MODERATE":
        if volatility in ["LOW", "VERY_LOW"]: macroRegime = "TREND_MOD_LOW_VOL"
        elif volatility == "MEDIUM": macroRegime = "TREND_MOD_MED_VOL"
        else: macroRegime = "TREND_MOD_HIGH_VOL"
    elif strength == "RANGING":
        if volatility in ["LOW", "VERY_LOW"]: macroRegime = "RANGE_LOW_VOL"
        elif volatility == "MEDIUM": macroRegime = "RANGE_MED_VOL"
        else: macroRegime = "RANGE_HIGH_VOL"
    else:
        if volatility == "HIGH": macroRegime = "WEAK_HIGH_VOL"
        elif volatility == "MEDIUM": macroRegime = "WEAK_MED_VOL"
        else: macroRegime = "WEAK_LOW_VOL"
    if isTransitioning and "_TRANSITION" not in macroRegime: macroRegime += "_TRANSITION"
    baseContext['macroRegime'] = macroRegime
    baseContext['isTransitioning'] = isTransitioning
    baseContext['details'] += f",Regime:{macroRegime}"
    return baseContext

def analyze_advanced_market_regime(trendContext, marketEntropyState):
    strength, volatility = trendContext.get('strength'), trendContext.get('volatility')
    entropy = marketEntropyState.get('state')
    probabilities = {'bullTrend': 0.25, 'bearTrend': 0.25, 'volatileRange': 0.25, 'quietRange': 0.25}
    if strength == 'STRONG' and volatility != 'HIGH' and entropy == 'ORDERLY':
        if 'BIG' in trendContext.get('direction', ''): probabilities = {'bullTrend': 0.8, 'bearTrend': 0.05, 'volatileRange': 0.1, 'quietRange': 0.05}
        else: probabilities = {'bullTrend': 0.05, 'bearTrend': 0.8, 'volatileRange': 0.1, 'quietRange': 0.05}
    elif strength == 'RANGING' and volatility == 'HIGH' and 'CHAOS' in entropy:
        probabilities = {'bullTrend': 0.1, 'bearTrend': 0.1, 'volatileRange': 0.7, 'quietRange': 0.1}
    elif strength == 'RANGING' and volatility == 'VERY_LOW':
        probabilities = {'bullTrend': 0.1, 'bearTrend': 0.1, 'volatileRange': 0.1, 'quietRange': 0.7}
    return {'probabilities': probabilities, 'details': f"Prob(B:{probabilities['bullTrend']:.2f},S:{probabilities['bearTrend']:.2f})"}

def update_shared_stats_for_next_cycle(latest_prediction, actual_result, history):
    global shared_stats_payload
    
    last_pred_correct = (latest_prediction.predicted_outcome == actual_result.actual_outcome)
    
    if not last_pred_correct:
        shared_stats_payload['consecutiveLosses'] += 1
    else:
        shared_stats_payload['consecutiveLosses'] = 0
    
    if shared_stats_payload['consecutiveLosses'] >= 3:
        shared_stats_payload['risk_aversion_factor'] = 0.5
    elif shared_stats_payload['consecutiveLosses'] == 0:
        shared_stats_payload['risk_aversion_factor'] = 1.0
    else:
        shared_stats_payload['risk_aversion_factor'] = 1.0 - (shared_stats_payload['consecutiveLosses'] / 10.0)

    shared_stats_payload['lastPeriodFull'] = latest_prediction.period
    shared_stats_payload['lastPredictedOutcome'] = latest_prediction.predicted_outcome
    shared_stats_payload['lastActualOutcome'] = actual_result.actual_outcome
    shared_stats_payload['lastFinalConfidence'] = latest_prediction.confidence
    shared_stats_payload['lastConfidenceLevel'] = latest_prediction.confidence_level
    shared_stats_payload['lastMacroRegime'] = latest_prediction.macro_regime
    last_signals = latest_prediction.contributing_signals
    if isinstance(last_signals, str):
        try: last_signals = json.loads(last_signals)
        except json.JSONDecodeError: last_signals = []
    shared_stats_payload['lastPredictionSignals'] = last_signals

    shared_stats_payload['lastConcentrationModeEngaged'] = getattr(latest_prediction, 'concentration_mode_engaged', False)
    shared_stats_payload['lastMarketEntropyState'] = getattr(latest_prediction, 'market_entropy_state', 'STABLE_MODERATE')
    shared_stats_payload['lastVolatilityRegime'] = getattr(latest_prediction, 'volatility_regime', 'MEDIUM')
    
    shared_stats_payload['longTermGlobalAccuracy'] = latest_prediction.prediction_quality_score
    
    if latest_prediction.predicted_outcome == shared_stats_payload.get('lastPredictedOutcome'):
        shared_stats_payload['consecutiveSamePredictions'] += 1
    else:
        shared_stats_payload['consecutiveSamePredictions'] = 1

    actuals = [get_big_small_from_number(p.actual_number) for p in history if hasattr(p, 'actual_number')]
    if actuals:
        streaks = {}
        for outcome_type in ['BIG', 'SMALL']:
            current_streak = 0
            max_streak = 0
            for outcome in actuals:
                if outcome == outcome_type:
                    current_streak += 1
                else:
                    max_streak = max(max_streak, current_streak)
                    current_streak = 0
            max_streak = max(max_streak, current_streak)
            streaks[outcome_type] = max_streak
        shared_stats_payload['historicalMaxStreaks'] = streaks


def calculateEngineWeights():
    global ENGINE_PERFORMANCE, ENGINES
    total_accuracy = 0
    weights = {}
    for engine in ENGINES:
        perf = ENGINE_PERFORMANCE[engine]
        if perf['wins'] + perf['losses'] > 0:
            perf['accuracy'] = perf['wins'] / (perf['wins'] + perf['losses'])
        else:
            perf['accuracy'] = 0.5
        total_accuracy += perf['accuracy']
    
    if total_accuracy > 0:
        for engine in ENGINES:
            weights[engine] = ENGINE_PERFORMANCE[engine]['accuracy'] / total_accuracy
    else:
        default_weight = 1.0 / len(ENGINES)
        for engine in ENGINES:
            weights[engine] = default_weight
            
    return weights

def getQStateImproved(last_outcome, current_streak, result_type):
    return f"{last_outcome}-{ 'W' if current_streak >= 0 else 'L' }-{abs(current_streak)}-{result_type}"

def updateQTable(state_name, action, reward, next_state, confidence):
    global qTable, learningRate, discountFactor
    if state_name not in qTable:
        qTable[state_name] = {}
    if action not in qTable[state_name]:
        qTable[state_name][action] = 0.0
    
    old_q = qTable[state_name][action]
    max_next_q = 0.0
    if next_state and next_state in qTable and qTable[next_state]:
        max_next_q = max(qTable[next_state].values())

    shaped_reward = reward + (reward * (confidence / 100.0) * 0.5)
    new_q = old_q + learningRate * (shaped_reward + discountFactor * max_next_q - old_q)
    qTable[state_name][action] = new_q
    # Note: In a full implementation, you would save this state, e.g., to a file.

def getQAction(state_name):
    global qTable, explorationRate, ENGINES
    if random.random() < explorationRate or state_name not in qTable or not qTable[state_name]:
        return random.choice(ENGINES)
    return max(qTable[state_name], key=qTable[state_name].get)

def update_engine_performance(engine_name, is_win):
    global ENGINE_PERFORMANCE
    if is_win:
        ENGINE_PERFORMANCE[engine_name]['wins'] += 1
        ENGINE_PERFORMANCE[engine_name]['lossStreak'] = 0
    else:
        ENGINE_PERFORMANCE[engine_name]['losses'] += 1
        ENGINE_PERFORMANCE[engine_name]['lossStreak'] += 1

def update_game_stats_for_q_learning(is_win, entry):
    global shared_stats_payload, qTable
    
    last_outcome_for_q = shared_stats_payload.get('lastActualOutcome')
    current_streak = shared_stats_payload.get('consecutiveLosses', 0) if not is_win else shared_stats_payload.get('consecutiveSamePredictions', 0)
    
    q_state = getQStateImproved(last_outcome_for_q, current_streak, 'bigsmall')
    
    reward = 1.0 if is_win else -1.0
    
    # Simulate next state based on hypothetical outcome of next cycle
    next_state_win = getQStateImproved(entry.get('prediction'), current_streak + 1 if is_win else 0, 'bigsmall')
    next_state_loss = getQStateImproved(get_opposite_outcome(entry.get('prediction')), 0 if is_win else current_streak + 1, 'bigsmall')
    
    updateQTable(q_state, entry.get('server'), reward, next_state_win, entry.get('probability'))
    update_engine_performance(entry.get('server'), is_win)

def quantum_ai_method(historical_data):
    """
    Python equivalent of QuantumAIMethod.
    """
    numbers = [int(d['actual_number']) for d in historical_data if d.get('actual_number') is not None]
    if len(numbers) < MIN_DATA_FOR_COMPLEX_AI or len(numbers) < 20:
        return {'result': "N/A", 'probability': 0}

    mean = sum(numbers[:20]) / 20.0
    std_dev = math.sqrt(sum((n - mean) ** 2 for n in numbers[:20]) / 19.0) if len(numbers) > 1 else 0

    if numbers[0] > (mean + 2 * std_dev):
        result = 'SMALL'
    elif numbers[0] < (mean - 2 * std_dev):
        result = 'BIG'
    else:
        result = 'BIG' if numbers[0] >= 5 else 'SMALL'
    
    probability = 75 + min(abs(numbers[0] - mean) / max(0.1, std_dev), 2) * 10
    
    return {'result': result, 'probability': round(probability)}

def neural_net_method(historical_data):
    """
    Python equivalent of NeuralNetMethod.
    Reuses the existing calculate_rsi function from prediction_engine.py.
    """
    numbers = [int(d['actual_number']) for d in historical_data if d.get('actual_number') is not None]
    if len(numbers) < MIN_DATA_FOR_COMPLEX_AI or len(numbers) < 15:
        return {'result': "N/A", 'probability': 0}

    rsi = calculate_rsi(numbers, 14)
    if rsi is None:
        return {'result': "N/A", 'probability': 0}
        
    if rsi > 70:
        result = 'SMALL'
    elif rsi < 30:
        result = 'BIG'
    else:
        result = 'BIG' if numbers[0] >= 5 else 'SMALL'
        
    probability = 70 + abs(rsi - 50) / 4.0
    return {'result': result, 'probability': round(probability)}

def fibonacci_method(historical_data):
    """
    Python equivalent of FibonacciMethod.
    """
    numbers = [int(d['actual_number']) for d in historical_data if d.get('actual_number') is not None]
    if len(numbers) < MIN_DATA_FOR_COMPLEX_AI or len(numbers) < 20:
        return {'result': "N/A", 'probability': 0}
        
    recent = numbers[:20]
    high = max(recent)
    low = min(recent)
    
    current_number = numbers[0]
    
    if high - low == 0:
        result = 'BIG' if current_number >= 5 else 'SMALL'
        return {'result': result, 'probability': 50}

    fib_level_50 = low + (high - low) * 0.5
    
    result = 'BIG' if current_number > fib_level_50 else 'SMALL'
    probability = 60 + (1 - (abs(current_number - fib_level_50) / (high - low))) * 30
    
    return {'result': result, 'probability': round(probability)}

def master_ai_method(historical_data):
    """
    Python equivalent of MasterAI_Method, acting as a fusion engine.
    """
    weights = calculateEngineWeights()

    predictions = {
        'QuantumAI': quantum_ai_method(historical_data),
        'NeuralNet': neural_net_method(historical_data),
        'Fibonacci': fibonacci_method(historical_data)
    }

    filtered_predictions = [
        {'name': name, 'result': p['result'], 'probability': p['probability'], 'weight': weights.get(name, 0)}
        for name, p in predictions.items() if p['result'] != 'N/A'
    ]
    
    if not filtered_predictions:
        return {'result': "N/A", 'probability': 0, 'isSureshot': False, 'levels': {'level1': 'All sub-engines failed.'}}

    first_prediction_result = filtered_predictions[0]['result']
    is_sureshot = all(p['result'] == first_prediction_result for p in filtered_predictions)

    votes = {'BIG': 0, 'SMALL': 0}
    total_confidence = 0
    total_weight = 0

    for p in filtered_predictions:
        if p['result'] in votes:
            adjusted_weight = p['weight']
            votes[p['result']] += adjusted_weight
            total_confidence += p['probability'] * adjusted_weight
            total_weight += adjusted_weight

    final_result = 'BIG' if votes['BIG'] > votes['SMALL'] else 'SMALL'
    final_confidence = round(total_confidence / total_weight) if total_weight > 0 else 50
    
    if is_sureshot:
        final_confidence = min(99, final_confidence + 15)
        
    return {'result': final_result, 'probability': max(0, min(99, final_confidence)), 'isSureshot': is_sureshot}


# --- Main Prediction Function ---
def ultraAIPredict(current_shared_history, shared_stats_payload=None):
    # defensive: ensure shared_stats_payload is a dict
    if 'shared_stats_payload' in locals() and shared_stats_payload is None:
        shared_stats_payload = {}

    global GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE, consecutiveHighConfLosses, reflexiveCorrectionActive, signalPerformance, REGIME_SIGNAL_PROFILES, driftDetector, MIN_ABSOLUTE_WEIGHT

    if not shared_stats_payload:
        shared_stats_payload = {'consecutiveSkips': 0, 'consecutiveLosses': 0}

    current_period_full = str(datetime.now().timestamp())
    
    time = get_current_ist_hour()
    primeTimeSession = get_prime_time_session(time.get('raw'))
    realTimeData = get_real_time_external_data()
    
    masterLogic = [f"LAYER 1: Initialization & Context (QAScore_v42.2.0, IST_Hr:{time.get('raw')}, ExtData:{realTimeData.get('reason') if realTimeData else 'Unavailable'})"]
    
    primeTimeAggression, primeTimeConfidence = 1.0, 1.0
    if primeTimeSession:
        masterLogic.append(f"LAYER 2: Prime Time Active ({primeTimeSession.get('session')})")
        primeTimeAggression = primeTimeSession.get('aggression')
        primeTimeConfidence = primeTimeSession.get('confidence')

    longTermGlobalAccuracy = shared_stats_payload.get('longTermGlobalAccuracy') or GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE
    if shared_stats_payload and isinstance(shared_stats_payload.get('longTermGlobalAccuracy'), (int, float)):
        GLOBAL_LONG_TERM_ACCURACY_FOR_LEARNING_RATE = shared_stats_payload.get('longTermGlobalAccuracy')

    trendContext = get_market_regime_and_trend_context(current_shared_history)
    stability = analyze_trend_stability(current_shared_history)
    marketEntropyAnalysis = analyze_market_entropy_state(current_shared_history, trendContext, stability)
    
    driftState = 'STABLE'
    if shared_stats_payload and shared_stats_payload.get('lastActualOutcome') is not None:
        lastPredictionWasCorrect = shared_stats_payload.get('lastActualOutcome') == shared_stats_payload.get('lastPredictedOutcome')
        driftState = detectConceptDrift(lastPredictionWasCorrect)

    masterLogic.append(f"LAYER 3: Pre-computation (Regime:{trendContext.get('macroRegime')}, Stability:{stability.get('isStable')}, Entropy:{marketEntropyAnalysis.get('state')}, Drift:{driftState})")

    bypassCheck = perform_six_level_bypass_check(current_shared_history, shared_stats_payload, stability, marketEntropyAnalysis, driftState)
    masterLogic.append(f"LAYER 4: 6-Level Bypass Validation (Passed: {bypassCheck.get('passed')})")

    risk_aversion_factor = shared_stats_payload.get('risk_aversion_factor', 1.0)
    if risk_aversion_factor < 1.0:
        masterLogic.append(f"RISK AVERSION ACTIVE (Factor: {risk_aversion_factor:.2f})")
    
    if not bypassCheck.get('passed'):
        masterLogic.append(f"BYPASS FAILED: {'; '.join(bypassCheck.get('reasons'))}. Forcing low-confidence prediction.")
        finalDecision = "BIG" if random.random() > 0.5 else "SMALL"
        return {
            'finalDecision': finalDecision, 'finalConfidence': 0.5, 'confidenceLevel': 1, 'isForcedPrediction': True,
            'overallLogic': ' -> '.join(masterLogic), 'source': "BypassValidation",
            'contributingSignals': [], 'currentMacroRegime': trendContext.get('macroRegime'), 'marketEntropyState': marketEntropyAnalysis.get('state'), 'predictionQualityScore': 0.01,
            'lastPredictedOutcome': finalDecision, 'lastFinalConfidence': 0.5, 'lastConfidenceLevel': 1, 'lastMacroRegime': trendContext.get('macroRegime'), 'lastPredictionSignals': [], 'lastVolatilityRegime': trendContext.get('volatility'),
            'trend_context': trendContext.get('details', {})
        }

    isReflexiveCorrection = checkForAnomalousPerformance(shared_stats_payload)
    concentrationModeEngaged = not stability.get('isStable') or isReflexiveCorrection or "CHAOS" in marketEntropyAnalysis.get('state', '') or driftState != 'STABLE'
    if concentrationModeEngaged: masterLogic.append("ConcentrationModeActive")

    currentVolatilityRegimeForPerf = trendContext.get('volatility')
    currentMacroRegime = trendContext.get('macroRegime')
    
    if shared_stats_payload and shared_stats_payload.get('lastPredictionSignals') and shared_stats_payload.get('lastActualOutcome'):
        update_signal_performance(
            shared_stats_payload.get('lastPredictionSignals'),
            shared_stats_payload.get('lastActualOutcome'),
            shared_stats_payload.get('lastPeriodFull'),
            shared_stats_payload.get('lastVolatilityRegime') or currentVolatilityRegimeForPerf,
            shared_stats_payload.get('lastFinalConfidence'),
            shared_stats_payload.get('lastConcentrationModeEngaged') or False,
            shared_stats_payload.get('lastMarketEntropyState') or "STABLE_MODERATE"
        )
        if shared_stats_payload.get('lastPredictedOutcome'):
            update_regime_profile_performance(shared_stats_payload.get('lastMacroRegime'), shared_stats_payload.get('lastActualOutcome'), shared_stats_payload.get('lastPredictedOutcome'))

    confirmed_history = [p for p in current_shared_history if p and p.get('actual_number') is not None]
    
    # --- Q-Learning & AI Method Selection ---
    if len(confirmed_history) > MIN_DATA_FOR_COMPLEX_AI:
        last_outcome_for_q = shared_stats_payload.get('lastActualOutcome')
        current_streak_for_q = shared_stats_payload.get('consecutiveLosses') if not last_outcome_for_q == shared_stats_payload.get('lastPredictedOutcome') else shared_stats_payload.get('consecutiveSamePredictions')
        
        q_state = getQStateImproved(last_outcome_for_q, current_streak_for_q, 'bigsmall')
        selected_engine_name = getQAction(q_state)

        # Update Q-table with the result of the previous prediction
        if shared_stats_payload.get('lastActualOutcome') and shared_stats_payload.get('lastPredictedOutcome'):
            is_win = shared_stats_payload.get('lastActualOutcome') == shared_stats_payload.get('lastPredictedOutcome')
            last_pred_entry = {'server': shared_stats_payload.get('lastPredictionSignals')[0]['source'] if shared_stats_payload.get('lastPredictionSignals') else 'N/A', 'probability': shared_stats_payload.get('lastFinalConfidence')*100}
            update_game_stats_for_q_learning(is_win, last_pred_entry)
        
        masterLogic.append(f"LAYER 5: MasterAI Engine Active -> Selected Engine: {selected_engine_name}")
        
        if selected_engine_name == 'MasterAI':
            final_pred_data = master_ai_method(confirmed_history)
        elif selected_engine_name == 'QuantumAI':
            final_pred_data = quantum_ai_method(confirmed_history)
        elif selected_engine_name == 'NeuralNet':
            final_pred_data = neural_net_method(confirmed_history)
        elif selected_engine_name == 'Fibonacci':
            final_pred_data = fibonacci_method(confirmed_history)
        else:
            final_pred_data = {'result': 'N/A', 'probability': 0}
            
        if final_pred_data['result'] != 'N/A':
            finalDecision = final_pred_data['result']
            finalConfidence = final_pred_data['probability'] / 100.0
            
            confidenceLevel = 1
            if finalConfidence > 0.65: confidenceLevel = 2
            if final_pred_data.get('isSureshot', False): confidenceLevel = 3
            
            output = {
                'finalDecision': finalDecision,
                'finalConfidence': finalConfidence,
                'confidenceLevel': confidenceLevel,
                'isForcedPrediction': False,
                'overallLogic': ' -> '.join(masterLogic) + f" -> Fusion via {selected_engine_name}",
                'source': selected_engine_name,
                'contributingSignals': [],
                'currentMacroRegime': 'MASTER_AI_FUSION',
                'marketEntropyState': 'MASTER_AI_FUSION',
                'predictionQualityScore': finalConfidence,
                'isPremium': True if confidenceLevel == 3 else False,
                'lastPredictedOutcome': finalDecision,
                'lastFinalConfidence': finalConfidence,
                'lastConfidenceLevel': confidenceLevel,
                'lastMacroRegime': 'MASTER_AI_FUSION',
                'lastPredictionSignals': [{'source': selected_engine_name, 'prediction': finalDecision, 'adjustedWeight': finalConfidence}],
                'lastConcentrationModeEngaged': concentrationModeEngaged,
                'lastMarketEntropyState': marketEntropyAnalysis.get('state'),
                'lastVolatilityRegime': trendContext.get('volatility'),
                'periodFull': current_period_full,
                'trend_context': trendContext,
                'strategy': 'MasterAI-Fusion',
                'is_sureshot': final_pred_data.get('isSureshot', False)
            }
            return output
            
    
    # ... (existing signal-based logic continues below if Q-learning fails or not enough data) ...
    
    signals = []
    currentRegimeProfile = REGIME_SIGNAL_PROFILES.get(currentMacroRegime) or REGIME_SIGNAL_PROFILES["DEFAULT"]
    regimeContextualAggression = (currentRegimeProfile.get('contextualAggression') or 1.0) * primeTimeAggression * risk_aversion_factor

    if isReflexiveCorrection or driftState == 'DRIFT': regimeContextualAggression *= 0.25
    elif concentrationModeEngaged: regimeContextualAggression *= 0.6

    masterLogic.append(f"LAYER 5: Signal Generation & Dynamic Adjustment (Aggression: {regimeContextualAggression:.2f})")
    
    def add_signal(fn, history_arg, signal_type, lookback_params, base_weight):
        if not ('all' in currentRegimeProfile.get('activeSignalTypes', []) or signal_type in currentRegimeProfile.get('activeSignalTypes', [])):
            return

        fn_args = [history_arg]
        if isinstance(lookback_params, dict): fn_args.extend(list(lookback_params.values()))
        if fn.__name__ in ['analyze_rsi', 'analyze_stochastic']: fn_args.append(trendContext.get('volatility'))
        if fn.__name__ == 'analyze_volatility_trend_fusion': fn_args.insert(1, marketEntropyAnalysis)
        if fn.__name__ == 'analyze_ml_model_signal':
            features = create_feature_set_for_ml(history_arg, trendContext, time)
            if not features: return
            fn_args = [features]
        if fn.__name__ in ['analyze_monte_carlo_signal', 'analyze_bayesian_inference', 'analyze_quantum_superposition_state']:
            fn_args = [signals]
        if fn.__name__ == 'analyze_adaptive_streaks':
            fn_args = [history_arg, shared_stats_payload]
        fn_args.append(base_weight)
        
        result = fn(*fn_args)

        if result and result.get('weight') and result.get('prediction'):
            trend_strength_mod = 1.0
            if trendContext.get('strength') == 'STRONG' and signal_type in ['trend', 'momentum']:
                trend_strength_mod = 1.25
            elif trendContext.get('strength') == 'RANGING' and signal_type in ['meanRev']:
                trend_strength_mod = 1.25
            result['adjustedWeight'] = get_dynamic_weight_adjustment(result['source'], result['weight'] * regimeContextualAggression * trend_strength_mod, current_period_full, currentVolatilityRegimeForPerf, current_shared_history)
            signals.append(result)

    add_signal(analyze_transitions, confirmed_history, 'pattern', {}, 0.05)
    add_signal(analyze_streaks, confirmed_history, 'pattern', {}, 0.045)
    add_signal(analyze_adaptive_streaks, confirmed_history, 'pattern', {}, 0.15)
    add_signal(analyze_alternating_patterns, confirmed_history, 'pattern', {}, 0.06)
    add_signal(analyze_heuristic_patterns, confirmed_history, 'pattern', {}, 0.15)
    add_signal(analyze_complex_patterns, confirmed_history, 'pattern', {}, 0.18) # Added
    add_signal(analyze_fibonacci_patterns, confirmed_history, 'pattern', {}, 0.10) # Added
    add_signal(analyze_weighted_historical, confirmed_history, 'pattern', {'weightDecayFactor': 0.9}, 0.05)
    add_signal(analyze_two_plus_one_patterns, confirmed_history, 'pattern', {}, 0.07)
    add_signal(analyze_double_patterns, confirmed_history, 'pattern', {}, 0.075)
    add_signal(analyze_mirror_patterns, confirmed_history, 'pattern', {}, 0.08)
    add_signal(analyze_rsi, confirmed_history, 'momentum', {'rsiPeriod': 28}, 0.08)
    add_signal(analyze_macd, confirmed_history, 'trend', {'shortPeriod': 12, 'longPeriod': 26, 'signalPeriod': 9}, 0.09)
    add_signal(analyze_bollinger_bands, confirmed_history, 'meanRev', {'period': 40, 'stdDevMultiplier': 2.1}, 0.07)
    add_signal(analyze_ichimoku_cloud, confirmed_history, 'trend', {'tenkanPeriod': 18, 'kijunPeriod': 52, 'senkouBPeriod': 104}, 0.14)
    add_signal(analyze_stochastic, confirmed_history, 'momentum', {'kPeriod': 14, 'dPeriod': 3, 'smoothK': 3}, 0.08)
    add_signal(analyze_ma_deviation, confirmed_history, 'meanRev', {'longMAPeriod': 20, 'normalizationPeriod': 30}, 0.10)
    add_signal(analyze_vwap_deviation, confirmed_history, 'meanRev', {'vwapPeriod': 15, 'normalizationPeriod': 20}, 0.09)
    add_signal(analyze_harmonic_potential, confirmed_history, 'pattern', {}, 0.06)
    add_signal(analyze_ngram_patterns, confirmed_history, 'pattern', {'n': 2}, 0.07)
    add_signal(analyze_ngram_patterns, confirmed_history, 'pattern', {'n': 3}, 0.06)
    add_signal(analyze_cyclical_patterns, confirmed_history, 'pattern', {'period': 10}, 0.05)
    add_signal(analyze_volatility_persistence, confirmed_history, 'volatility', {'period': 10}, 0.07)
    add_signal(analyze_fractal_dimension, confirmed_history, 'volatility', {'period': 14}, 0.04)
    add_signal(analyze_entropy_signal, confirmed_history, 'volatility', {'period': 15}, 0.06)
    add_signal(analyze_volatility_breakout, confirmed_history, 'volatility', {}, 0.07)
    add_signal(analyze_waveform_patterns, confirmed_history, 'pattern', {}, 0.035)
    add_signal(analyze_phase_space, confirmed_history, 'pattern', {}, 0.04)
    add_signal(analyze_quantum_tunneling, confirmed_history, 'volatility', {}, 0.055)
    add_signal(analyze_entanglement, confirmed_history, 'probabilistic', {'lag': 3}, 0.04)
    add_signal(analyze_entanglement, confirmed_history, 'probabilistic', {'lag': 5}, 0.03)

    signals = analyze_and_correct_prediction_loops(shared_stats_payload, signals) # Upgraded loop protection
    if shared_stats_payload.get('consecutiveLosses', 0) >= 2:
        masterLogic.append(f"LoopCorrection(L:{shared_stats_payload.get('consecutiveLosses', 0)})")

    masterLogic.append("LAYER 6: Meta-Signal Generation")
    add_signal(analyze_volatility_trend_fusion, trendContext, 'fusion', {}, 0.25)
    add_signal(analyze_ml_model_signal, confirmed_history, 'ml', {}, 0.40)
    add_signal(analyze_bayesian_inference, signals, 'probabilistic', {}, 0.15)
    add_signal(analyze_monte_carlo_signal, signals, 'probabilistic', {}, 0.18)
    
    consensus = analyze_prediction_consensus(signals, trendContext)
    masterLogic.append(f"LAYER 7: Cross-Signal Consensus Check (Factor:{consensus.get('factor'):.2f})")
    superpositionSignal = analyze_quantum_superposition_state(signals, consensus, 0.22)
    if superpositionSignal:
        superpositionSignal['adjustedWeight'] = get_dynamic_weight_adjustment(superpositionSignal.get('source'), superpositionSignal.get('weight'), current_period_full, currentVolatilityRegimeForPerf, current_shared_history)
        signals.append(superpositionSignal)
    
    validSignals = [s for s in signals if s.get('prediction') and s.get('adjustedWeight', 0) > MIN_ABSOLUTE_WEIGHT]
    masterLogic.append(f"ValidSignals({len(validSignals)}/{len(signals)})")
    
    bigScore, smallScore = 0, 0
    for signal in validSignals:
        if signal.get('prediction') == "BIG": bigScore += signal.get('adjustedWeight', 0)
        elif signal.get('prediction') == "SMALL": smallScore += signal.get('adjustedWeight', 0)
    
    masterLogic.append("LAYER 8: Advanced Probabilistic Regime Adjustment")
    advancedRegime = analyze_advanced_market_regime(trendContext, marketEntropyAnalysis)
    bigScore *= (1 + advancedRegime.get('probabilities', {}).get('bullTrend', 0) - advancedRegime.get('probabilities', {}).get('bearTrend', 0))
    smallScore *= (1 + advancedRegime.get('probabilities', {}).get('bearTrend', 0) - advancedRegime.get('probabilities', {}).get('bullTrend', 0))
    bigScore *= consensus.get('factor', 1.0)
    smallScore *= (2.0 - consensus.get('factor', 1.0))
    
    totalScore = bigScore + smallScore
    finalDecision = "BIG" if totalScore > 0 and bigScore >= smallScore else ("SMALL" if totalScore > 0 and smallScore > bigScore else ("BIG" if random.random() > 0.5 else "SMALL"))
    finalConfidence = max(bigScore, smallScore) / totalScore if totalScore > 0 else 0.5
    
    if realTimeData:
        finalConfidence = 0.5 + (finalConfidence - 0.5) * primeTimeConfidence * realTimeData.get('factor', 1.0)
    else:
        finalConfidence = 0.5 + (finalConfidence - 0.5) * primeTimeConfidence

    signalConsistency = analyze_signal_consistency(validSignals, trendContext)
    pathConfluence = analyze_path_confluence_strength(validSignals, finalDecision)
    predictiveDivergence = analyze_predictive_divergence(validSignals, finalDecision)
    masterLogic.append(f"LAYER 9: Signal Consistency & Path Confluence (Consistency:{signalConsistency.get('score'):.2f}, Confluence:{pathConfluence.get('score'):.2f}, Divergence:{predictiveDivergence.get('divergence_score'):.2f})")
    
    uncertainty = calculate_uncertainty_score(trendContext, stability, marketEntropyAnalysis, signalConsistency, pathConfluence, longTermGlobalAccuracy, isReflexiveCorrection, driftState, predictiveDivergence)
    uncertaintyFactor = 1.0 - min(1.0, uncertainty.get('score', 0) / 120.0)
    finalConfidence = 0.5 + (finalConfidence - 0.5) * uncertaintyFactor
    
    finalConfidence, diversity_logic = apply_prediction_diversity_logic(finalConfidence, shared_stats_payload)
    if diversity_logic:
        masterLogic.append(diversity_logic)

    finalConfidence *= risk_aversion_factor
    masterLogic.append(f"LAYER 10: Uncertainty Modulation & Final Calibration (Uncertainty Score:{uncertainty.get('score'):.0f}, Factor:{uncertaintyFactor:.2f}; Reasons:{uncertainty.get('reasons')})")
    
    pqs = 0.5
    pqs += (signalConsistency.get('score', 0) - 0.5) * 0.4
    pqs += pathConfluence.get('score', 0) * 1.2
    pqs = max(0.01, min(0.99, pqs - (uncertainty.get('score', 0) / 500)))
    masterLogic.append(f"PQS:{pqs:.3f}")

    highConfThreshold, medConfThreshold = 0.78, 0.65
    highPqsThreshold, medPqsThreshold = 0.75, 0.60
    if primeTimeSession:
        highConfThreshold, medConfThreshold = 0.72, 0.60
        highPqsThreshold, medPqsThreshold = 0.70, 0.55
    
    confidenceLevel = 1
    if finalConfidence > medConfThreshold and pqs > medPqsThreshold: confidenceLevel = 2
    if finalConfidence > highConfThreshold and pqs > highPqsThreshold: confidenceLevel = 3
    
    uncertaintyThreshold = 65 if isReflexiveCorrection or driftState == 'DRIFT' else 95
    isForced = uncertainty.get('score', 0) >= uncertaintyThreshold or pqs < 0.20
    
    shouldSkip = (pqs < 0.40 or uncertainty.get('score') > 85)
    
    if shouldSkip and shared_stats_payload.get('consecutiveSkips', 0) < MAX_CONSECUTIVE_SKIPS:
        finalDecision = "SKIP"
        confidenceLevel = 0
        finalConfidence = 0.5
        isForced = True
        masterLogic.append(f"SKIP(PQS:{pqs:.2f},U:{uncertainty.get('score'):.0f})")
        shared_stats_payload['consecutiveSkips'] = shared_stats_payload.get('consecutiveSkips', 0) + 1
    else:
        shared_stats_payload['consecutiveSkips'] = 0
        if isForced:
            confidenceLevel = 1
            finalConfidence = 0.5 + (random.random() - 0.5) * 0.02
            masterLogic.append(f"FORCED_PREDICTION(Uncertainty:{uncertainty.get('score', 0)}/{uncertaintyThreshold},PQS:{pqs})")

    final_strategy = determine_prediction_strategy(validSignals, finalDecision)
    masterLogic.append(f"Strategy:{final_strategy}")

    output = {
        'finalDecision': finalDecision,
        'finalConfidence': finalConfidence,
        'confidenceLevel': confidenceLevel,
        'isForcedPrediction': isForced,
        'overallLogic': ' -> '.join(masterLogic),
        'source': "RealTimeFusionV42.2",
        'contributingSignals': [{'source': s.get('source'), 'prediction': s.get('prediction'), 'weight': f"{s.get('adjustedWeight', 0):.5f}"} for s in validSignals[:15]],
        'currentMacroRegime': currentMacroRegime,
        'marketEntropyState': marketEntropyAnalysis.get('state'),
        'predictionQualityScore': pqs,
        'isPremium': True if confidenceLevel == 3 else False,
        'lastPredictedOutcome': finalDecision,
        'lastFinalConfidence': finalConfidence,
        'lastConfidenceLevel': confidenceLevel,
        'lastMacroRegime': currentMacroRegime,
        'lastPredictionSignals': [{'source': s.get('source'), 'prediction': s.get('prediction'), 'weight': s.get('adjustedWeight'), 'isOnProbation': s.get('isOnProbation', False)} for s in validSignals],
        'lastConcentrationModeEngaged': concentrationModeEngaged,
        'lastMarketEntropyState': marketEntropyAnalysis.get('state'),
        'lastVolatilityRegime': trendContext.get('volatility'),
        'periodFull': current_period_full,
        'trend_context': trendContext,
        'strategy': final_strategy,
        'is_sureshot': False
    }
    
    return output

# --- AUTO-ADDED: multi-step Monte Carlo support & Q-learning strategy integration ---
try:
    from multi_step import compute_multilevel_probs, simulate_next_k, build_transition_probs
    from q_learning_strategy import QLearner
except Exception as _e:
    # helpers not available in import path
    def compute_multilevel_probs(*args, **kwargs):
        return {'level_probs':[], 'p_win_k':0.0, 'target_label':None, 'meta':{}}
    def simulate_next_k(*args, **kwargs):
        return 0.0
    class QLearner:
        def __init__(self,*a,**k): pass
        def choose(self,*a,**k): return 'bet_level1'

def compute_p_win_3_and_attach(history, engine_probs, sims=1500, mix=0.6):
    '''Compute multi-step probabilities and return a dict containing p_win_3 and level_probs.'''
    try:
        res = compute_multilevel_probs(history, engine_probs, k=3, sims=sims, mix=mix)
        return {
            'p_win_3': res.get('p_win_k', 0.0),
            'level_probs': res.get('level_probs', []),
            'target_label': res.get('target_label'),
            'meta': res.get('meta', {})
        }
    except Exception as e:
        return {'p_win_3': 0.0, 'level_probs': [], 'target_label': engine_probs and max(engine_probs, key=engine_probs.get) or None, 'meta': {'error':str(e)}}

# Wrapper: call existing ultraAIPredict() and enrich output with p_win_3 computed via Monte Carlo multi-step.
def ultraAIPredict_enriched(*args, sims=1500, mix=0.6, qtable_path='q_table.json', **kwargs):
    '''
    Calls the original ultraAIPredict (should be defined in this module) and enriches its output.
    Returns same dict but with additional keys:
      - p_win_3 : probability of at least one correct outcome within next 3 levels for the chosen target
      - level_probs : list of per-level first-hit probabilities
      - multi_meta : metadata about simulation
      - suggested_strategy : chosen strategy from Q-Learner ('bet_level1','staged','hedge')
    '''
    if 'ultraAIPredict' not in globals():
        raise RuntimeError("ultraAIPredict not defined in this module; please ensure original function exists.")
    out = ultraAIPredict(*args, **kwargs)
    try:
        # Expect out contains 'engine_probs' (dict) and 'history' or accept kwargs['history']
        engine_probs = out.get('engine_probs') or kwargs.get('engine_probs') or {}
        history = out.get('history') or kwargs.get('history') or []
        if engine_probs and history:
            multi = compute_p_win_3_and_attach(history, engine_probs, sims=sims, mix=mix)
            out.update({
                'p_win_3': multi.get('p_win_3', 0.0),
                'level_probs': multi.get('level_probs', []),
                'multi_meta': multi.get('meta', {})
            })
            # simple pattern_key: use top-2 history as pattern identifier
            pattern_key = ''.join(history[-2:]) if len(history)>=2 else ''.join(history) if history else 'NONE'
            # integrate Q-Learner to suggest a strategy
            q = QLearner(filepath=qtable_path)
            loss_streak = kwargs.get('loss_streak', 0)
            suggested = q.choose(loss_streak, pattern_key or 'NONE')
            out['suggested_strategy'] = suggested
        else:
            out['p_win_3'] = out.get('p_win_3', 0.0)
    except Exception as e:
        out['p_win_3_error'] = str(e)
    return out

# End of AUTO-ADDED
