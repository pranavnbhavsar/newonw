
"""math_core.py
Advanced math and logic module for MrPerfect V5 prediction engine.
This file implements many of the math rules, scoring, and decision helpers
we discussed (trend strength, confidence combining, volatility, skip logic,
mirror/trap detection, weighted voting, LM weighting, emergency corrector, etc.)

Usage:
    from math_core import MathCore
    mc = MathCore(history_list)  # history_list: list of outcomes like ['BIG','SMALL','BIG',...]
    decision = mc.make_decision(engine_probs, lm_response=None, loss_streak=0)
    # engine_probs: list of tuples [(engine_name, prob_big, prob_small, engine_confidence), ...]
    # returns dict with prediction, final_conf, reason, diagnostics...
"""

from collections import Counter, deque
import math
import statistics
import itertools
import time

# Helper conversions for uniform representation
BIG = 'BIG'
SMALL = 'SMALL'

def last_n(history, n):
    return history[-n:] if len(history) >= n else history[:]

class MathCore:
    def __init__(self, history=None):
        # history should be a list of 'BIG'/'SMALL' or 'R'/'G' etc.
        self.history = history[:] if history else []
        # Keep local cache of last 1000
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        self.engine_weights = {}
        self.default_weights = {
            'logic_engine': 0.35,
            'pattern_engine': 0.25,
            'lm_engine': 0.20,
            'guarantee_engine': 0.20
        }
        self.min_confidence_to_predict = 0.55

    # -- Basic stats
    def long_term_ratio(self, window=200):
        h = last_n(self.history, window)
        if not h:
            return 0.5
        big = sum(1 for x in h if x == BIG)
        return big / len(h)

    def trend_strength(self, window=10):
        h = last_n(self.history, window)
        if not h:
            return 0.0
        most_common = Counter(h).most_common(1)[0][1] / len(h)
        return most_common

    def streak_length(self):
        if not self.history:
            return 0
        cnt = 1
        last = self.history[-1]
        for v in reversed(self.history[:-1]):
            if v == last:
                cnt += 1
            else:
                break
        return cnt

    def volatility(self, window=20):
        h = last_n(self.history, window)
        # convert BIG/SMALL to 1/0 numeric series for std dev
        if not h:
            return 0.0
        nums = [1 if x == BIG else 0 for x in h]
        if len(nums) < 2:
            return 0.0
        return statistics.pstdev(nums)

    # -- Pattern detectors
    def is_trap_pattern(self, window=12, threshold=0.8):
        h = last_n(self.history, window)
        if len(h) < 6:
            return False, 0.0
        alternates = sum(1 for a,b in zip(h, h[1:]) if a != b)
        trap_index = alternates / max(1, (len(h)-1))
        return (trap_index > threshold), trap_index

    def mirror_score(self, length=6):
        if len(self.history) < length*2:
            return 0.0
        a = self.history[-length:]
        b = self.history[-length*2:-length]
        # compare a to reversed(b)
        matches = sum(1 for x,y in zip(a, reversed(b)) if x==y)
        return matches / length

    def repeating_cycle_score(self, block=3):
        # check if last (block*3) can be divided into repeating blocks
        L = block*3
        if len(self.history) < L:
            return 0.0
        segment = self.history[-L:]
        # try to find smallest repeating unit
        for size in range(1, block+1):
            unit = segment[:size]
            repeated = unit * (L//size)
            if repeated == segment:
                return 1.0
        return 0.0

    # -- Confidence and weighting
    def combine_confidences(self, engine_conf_list):
        # engine_conf_list: list of floats (0..1)
        if not engine_conf_list:
            return 0.0
        # simple average with downweighting of outliers
        avg = sum(engine_conf_list)/len(engine_conf_list)
        # reduce by stddev fractionally
        if len(engine_conf_list) > 1:
            sd = statistics.pstdev(engine_conf_list)
            adj = max(0.0, avg - sd*0.3)
            return adj
        return avg

    def time_based_lm_weight(self, lm_time_s):
        if lm_time_s is None:
            return 0.0
        if lm_time_s < 5:
            return 1.0
        if lm_time_s <= 10:
            return 0.8
        return 0.0

    # -- Decision core
    def compute_engine_scores(self, engine_probs):
        # engine_probs: list of dicts {name, prob_big, prob_small, conf, response_time}
        # Return weighted big_score, small_score, individual contributions
        big_score = 0.0
        small_score = 0.0
        contributions = {}
        for e in engine_probs:
            name = e.get('name')
            pb = float(e.get('prob_big', 0.5))
            ps = float(e.get('prob_small', 1-pb))
            conf = float(e.get('conf', 0.5))
            # engine weight fallback to default mapping by name prefix
            weight = self.engine_weights.get(name, None)
            if weight is None:
                # pick a default bucket
                if 'logic' in name.lower():
                    weight = self.default_weights['logic_engine']
                elif 'pattern' in name.lower():
                    weight = self.default_weights['pattern_engine']
                elif 'lm' in name.lower():
                    weight = self.default_weights['lm_engine']
                elif 'guarantee' in name.lower():
                    weight = self.default_weights['guarantee_engine']
                else:
                    weight = 0.1
            # time-based adjustment for LM
            if 'lm' in name.lower() and 'response_time' in e:
                weight *= self.time_based_lm_weight(e.get('response_time'))
            # contribution scaled by confidence
            big_contrib = weight * conf * pb
            small_contrib = weight * conf * ps
            big_score += big_contrib
            small_score += small_contrib
            contributions[name] = {'big': big_contrib, 'small': small_contrib, 'weight': weight, 'conf': conf}
        return big_score, small_score, contributions

    def final_confidence(self, big_score, small_score, engine_contribs, loss_streak=0):
        total = big_score + small_score
        if total <= 0:
            return 0.0
        base_conf = max(big_score, small_score) / total
        # apply decay for recent losses
        adj = base_conf * (0.95 ** loss_streak)
        # stability penalty
        vol = self.volatility()
        stability = max(0.0, 1 - vol)
        adj *= stability
        return adj

    def make_decision(self, engine_probs, lm_response=None, loss_streak=0):
        """Main decision API.
        engine_probs: list of dicts {name, prob_big, prob_small, conf, response_time}
        lm_response: optional dict {'prob_big':..., 'prob_small':..., 'conf':..., 'response_time':...}
        returns dict with prediction, final_conf, reason, diagnostics
        """
        # compute trap/mirror/streak diagnostics first
        trap, trap_index = self.is_trap_pattern()
        mirror = self.mirror_score()
        streak = self.streak_length()
        lt_ratio = self.long_term_ratio()

        # aggregate engines (include lm_response into engine_probs if provided)
        engines = list(engine_probs)
        if lm_response:
            engines = engines + [dict(name='LM_engine', **lm_response)]

        big_score, small_score, contribs = self.compute_engine_scores(engines)
        final_conf = self.final_confidence(big_score, small_score, contribs, loss_streak=loss_streak)

        # emergency corrector logic
        if loss_streak >= 2:
            reason = 'emergency_corrector: loss_streak >=2, anti-pattern enforced'
            pred = SMALL if self.history and self.history[-1] == BIG else BIG
            final_conf = max(final_conf, 0.6)
            return {'prediction': pred, 'final_conf': final_conf, 'reason': reason, 'diagnostics': {'streak': streak, 'trap_index': trap_index, 'mirror': mirror, 'lt_ratio': lt_ratio, 'contribs': contribs}}

        # skip rules
        if final_conf < self.min_confidence_to_predict:
            return {'prediction': None, 'final_conf': final_conf, 'reason': 'skip_low_confidence', 'diagnostics': {'streak': streak, 'trap_index': trap_index, 'mirror': mirror, 'lt_ratio': lt_ratio, 'contribs': contribs}}

        # choose prediction based on scores
        prediction = BIG if big_score > small_score else SMALL
        reason = 'consensus_weighted'
        # apply anti-pattern if trap detected
        if trap and trap_index > 0.8:
            # invert prediction to break trap
            prediction = SMALL if prediction == BIG else BIG
            reason = 'trap_detected_inverted'

        return {'prediction': prediction, 'final_conf': final_conf, 'reason': reason, 'diagnostics': {'streak': streak, 'trap_index': trap_index, 'mirror': mirror, 'lt_ratio': lt_ratio, 'contribs': contribs}}

    # Utility to update history safely
    def push(self, result):
        if result not in (BIG, SMALL):
            # attempt normalization
            if result in ('R','RED','G','GREEN'):
                # map red/green to BIG/SMALL arbitrarily if needed
                result = BIG if result in ('R','RED') else SMALL
            else:
                return False
        self.history.append(result)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        return True

# Example quick test when run as script
if __name__ == '__main__':
    mc = MathCore(['BIG','BIG','BIG','SMALL','BIG','BIG','BIG','BIG','BIG','BIG'])
    engines = [
        {'name':'logic_engine_v1','prob_big':0.8,'prob_small':0.2,'conf':0.9},
        {'name':'pattern_engine_v2','prob_big':0.7,'prob_small':0.3,'conf':0.85},
        {'name':'guarantee_engine','prob_big':0.6,'prob_small':0.4,'conf':0.8}
    ]
    print(mc.make_decision(engines, lm_response={'prob_big':0.65,'prob_small':0.35,'conf':0.7,'response_time':2.0}, loss_streak=0))
