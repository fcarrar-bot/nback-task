#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-Back Demo - Quick Version (~2-3 min)
======================================
2-back only, 15 trials, for testing the setup.
"""

from psychopy import visual, core, event, gui
import os
import random
import numpy as np
from datetime import datetime

# ============================================================================
# SETTINGS
# ============================================================================

STIMULUS_DURATION = 0.5      # 500ms
ISI_DURATION = 1.5           # 1500ms (faster for demo)
TRIALS = 15                  # Short
PRACTICE_TRIALS = 5
TARGET_RATIO = 0.30
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'U']
N_BACK = 2

# ============================================================================
# FUNCTIONS
# ============================================================================

def generate_sequence(n_trials, n_back, target_ratio, letters):
    sequence = []
    is_target = []
    n_targets = int(n_trials * target_ratio)
    
    for i in range(n_back):
        sequence.append(random.choice(letters))
        is_target.append(False)
    
    remaining = n_trials - n_back
    target_positions = random.sample(range(n_back, n_trials), min(n_targets, remaining))
    
    for i in range(n_back, n_trials):
        if i in target_positions:
            sequence.append(sequence[i - n_back])
            is_target.append(True)
        else:
            available = [l for l in letters if l != sequence[i - n_back]]
            sequence.append(random.choice(available) if available else random.choice(letters))
            is_target.append(False)
    
    return sequence, is_target


def calculate_dprime(hits, misses, false_alarms, correct_rejections):
    from scipy import stats
    
    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    
    if n_targets == 0 or n_nontargets == 0:
        return float('nan')
    
    hit_rate = (hits + 0.5) / (n_targets + 1)
    fa_rate = (false_alarms + 0.5) / (n_nontargets + 1)
    
    return stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)


def run_block(win, letter_stim, fixation, n_trials, is_practice=False):
    sequence, targets = generate_sequence(n_trials, N_BACK, TARGET_RATIO, LETTERS)
    
    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0, 'rts': []}
    clock = core.Clock()
    
    for letter, is_target in zip(sequence, targets):
        if event.getKeys(['escape']):
            return None
        
        letter_stim.text = letter
        letter_stim.draw()
        win.flip()
        
        clock.reset()
        response = None
        rt = None
        
        while clock.getTime() < STIMULUS_DURATION:
            keys = event.getKeys(['space'], timeStamped=clock)
            if keys and response is None:
                response = True
                rt = keys[0][1]
        
        fixation.draw()
        win.flip()
        
        start = clock.getTime()
        while clock.getTime() < start + ISI_DURATION:
            keys = event.getKeys(['space'], timeStamped=clock)
            if keys and response is None:
                response = True
                rt = keys[0][1]
        
        if not is_practice:
            if is_target:
                if response:
                    results['hits'] += 1
                    results['rts'].append(rt)
                else:
                    results['misses'] += 1
            else:
                if response:
                    results['false_alarms'] += 1
                else:
                    results['correct_rejections'] += 1
    
    return results


def main():
    # Participant dialog
    dlg = gui.Dlg(title='N-Back Demo')
    dlg.addField('Participant ID:', 'demo')
    info = dlg.show()
    
    if not dlg.OK:
        core.quit()
    
    participant_id = info[0]
    
    # Setup window
    win = visual.Window(fullscr=True, color='black', units='height')
    letter_stim = visual.TextStim(win, text='', height=0.15, color='white')
    fixation = visual.TextStim(win, text='+', height=0.05, color='white')
    instruction = visual.TextStim(win, text='', height=0.03, color='white', wrapWidth=0.8)
    
    try:
        # Instructions
        instruction.text = """
2-BACK DEMO (~2-3 min)

Press SPACE when the current letter matches
the one shown 2 trials ago.

Example: A...B...A → press SPACE on the 2nd A

Press SPACE to start practice.
"""
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Practice
        run_block(win, letter_stim, fixation, PRACTICE_TRIALS, is_practice=True)
        
        instruction.text = "Practice done!\n\nPress SPACE for the real task."
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Main task
        results = run_block(win, letter_stim, fixation, TRIALS, is_practice=False)
        
        if results is None:
            win.close()
            core.quit()
        
        # Calculate metrics
        total = results['hits'] + results['misses'] + results['false_alarms'] + results['correct_rejections']
        accuracy = (results['hits'] + results['correct_rejections']) / total if total > 0 else 0
        dprime = calculate_dprime(results['hits'], results['misses'], results['false_alarms'], results['correct_rejections'])
        mean_rt = np.mean(results['rts']) * 1000 if results['rts'] else float('nan')
        
        # Save data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(data_dir, f'{participant_id}_demo_{timestamp}.csv')
        
        with open(filename, 'w') as f:
            f.write('participant_id,mode,n_back,trials,hits,misses,false_alarms,correct_rejections,accuracy,dprime,mean_rt_ms\n')
            f.write(f'{participant_id},demo,{N_BACK},{TRIALS},{results["hits"]},{results["misses"]},{results["false_alarms"]},{results["correct_rejections"]},{accuracy:.3f},{dprime:.2f},{mean_rt:.0f}\n')
        
        # Show results
        instruction.text = f"""
DEMO COMPLETE!

Accuracy: {accuracy*100:.0f}%
d-prime: {dprime:.2f}
Mean RT: {mean_rt:.0f} ms

Hits: {results['hits']}, Misses: {results['misses']}
False Alarms: {results['false_alarms']}

Data saved to: data/

Press any key to exit.
"""
        instruction.draw()
        win.flip()
        event.waitKeys()
        
    finally:
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
