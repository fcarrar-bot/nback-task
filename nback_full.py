#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-Back Full Version (~15 min)
=============================
2-back task with 48 trials × 2 blocks for proper measurement.
"""

from psychopy import visual, core, event, gui
import os
import random
import numpy as np
from datetime import datetime
import pandas as pd

# ============================================================================
# SETTINGS
# ============================================================================

STIMULUS_DURATION = 0.5      # 500ms
ISI_DURATION = 2.0           # 2000ms
TRIALS_PER_BLOCK = 48
BLOCKS = 2
PRACTICE_TRIALS = 10
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


def run_block(win, letter_stim, fixation, n_trials, trial_data, block_num, participant_id, is_practice=False):
    sequence, targets = generate_sequence(n_trials, N_BACK, TARGET_RATIO, LETTERS)
    
    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0, 'rts': []}
    clock = core.Clock()
    
    for trial_num, (letter, is_target) in enumerate(zip(sequence, targets)):
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
        
        # Determine response type
        if is_target:
            if response:
                response_type = 'hit'
                results['hits'] += 1
                results['rts'].append(rt)
            else:
                response_type = 'miss'
                results['misses'] += 1
        else:
            if response:
                response_type = 'false_alarm'
                results['false_alarms'] += 1
            else:
                response_type = 'correct_rejection'
                results['correct_rejections'] += 1
        
        # Store trial data
        if not is_practice:
            trial_data.append({
                'participant_id': participant_id,
                'block': block_num,
                'trial': trial_num + 1,
                'letter': letter,
                'is_target': is_target,
                'responded': response is not None,
                'response_type': response_type,
                'correct': response_type in ['hit', 'correct_rejection'],
                'rt': rt if rt else np.nan,
                'timestamp': datetime.now().isoformat()
            })
    
    return results


def main():
    # Participant dialog
    dlg = gui.Dlg(title='N-Back Task (Full)')
    dlg.addField('Participant ID:', '')
    dlg.addField('Session:', 1)
    info = dlg.show()
    
    if not dlg.OK:
        core.quit()
    
    participant_id = info[0]
    session = int(info[1])
    
    if not participant_id:
        print("Participant ID required")
        core.quit()
    
    # Setup window
    win = visual.Window(fullscr=True, color='black', units='height')
    letter_stim = visual.TextStim(win, text='', height=0.15, color='white')
    fixation = visual.TextStim(win, text='+', height=0.05, color='white')
    instruction = visual.TextStim(win, text='', height=0.03, color='white', wrapWidth=0.8)
    
    trial_data = []
    all_results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0, 'rts': []}
    
    try:
        # Welcome
        instruction.text = f"""
2-BACK WORKING MEMORY TASK

You will see letters appear one at a time.

Press SPACE when the current letter matches
the one shown 2 trials ago.

Example: A...B...A → press SPACE on the 2nd A

You will complete:
- Practice ({PRACTICE_TRIALS} trials)
- {BLOCKS} blocks × {TRIALS_PER_BLOCK} trials

Total time: ~15 minutes

Press SPACE to begin.
"""
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Practice
        instruction.text = "PRACTICE\n\nPress SPACE to start."
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        run_block(win, letter_stim, fixation, PRACTICE_TRIALS, [], 0, participant_id, is_practice=True)
        
        instruction.text = "Practice complete!\n\nThe main task will now begin.\n\nPress SPACE when ready."
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Main blocks
        for block in range(1, BLOCKS + 1):
            instruction.text = f"BLOCK {block} of {BLOCKS}\n\nPress SPACE to start."
            instruction.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            results = run_block(win, letter_stim, fixation, TRIALS_PER_BLOCK, trial_data, block, participant_id, is_practice=False)
            
            if results is None:
                win.close()
                core.quit()
            
            # Accumulate results
            for key in ['hits', 'misses', 'false_alarms', 'correct_rejections']:
                all_results[key] += results[key]
            all_results['rts'].extend(results['rts'])
            
            # Break between blocks
            if block < BLOCKS:
                instruction.text = f"Block {block} complete.\n\nTake a short break if needed.\n\nPress SPACE to continue."
                instruction.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
        
        # Calculate final metrics
        total = all_results['hits'] + all_results['misses'] + all_results['false_alarms'] + all_results['correct_rejections']
        accuracy = (all_results['hits'] + all_results['correct_rejections']) / total if total > 0 else 0
        dprime = calculate_dprime(all_results['hits'], all_results['misses'], all_results['false_alarms'], all_results['correct_rejections'])
        mean_rt = np.mean(all_results['rts']) * 1000 if all_results['rts'] else float('nan')
        
        # Save data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Trial-by-trial
        trial_file = os.path.join(data_dir, f'{participant_id}_s{session}_trials_{timestamp}.csv')
        pd.DataFrame(trial_data).to_csv(trial_file, index=False)
        
        # Summary
        summary = {
            'participant_id': participant_id,
            'session': session,
            'n_back': N_BACK,
            'total_trials': total,
            'hits': all_results['hits'],
            'misses': all_results['misses'],
            'false_alarms': all_results['false_alarms'],
            'correct_rejections': all_results['correct_rejections'],
            'accuracy': round(accuracy, 3),
            'dprime': round(dprime, 2),
            'mean_rt_ms': round(mean_rt, 0) if not np.isnan(mean_rt) else np.nan,
            'timestamp': timestamp
        }
        
        summary_file = os.path.join(data_dir, f'{participant_id}_s{session}_summary_{timestamp}.csv')
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        
        # Append to aggregated file
        agg_file = os.path.join(data_dir, 'all_participants.csv')
        if os.path.exists(agg_file):
            existing = pd.read_csv(agg_file)
            combined = pd.concat([existing, pd.DataFrame([summary])], ignore_index=True)
            combined.to_csv(agg_file, index=False)
        else:
            pd.DataFrame([summary]).to_csv(agg_file, index=False)
        
        # Exclusion check
        exclude = dprime < 1.1 or accuracy < 0.60
        exclude_text = "\n⚠️ BELOW THRESHOLD (d' < 1.1 or acc < 60%)" if exclude else "\n✓ Within normal range"
        
        # Show results
        instruction.text = f"""
TASK COMPLETE!

RESULTS:
  Accuracy: {accuracy*100:.1f}%
  d-prime: {dprime:.2f}
  Mean RT: {mean_rt:.0f} ms

  Hits: {all_results['hits']}, Misses: {all_results['misses']}
  False Alarms: {all_results['false_alarms']}
{exclude_text}

Data saved to: data/

Press any key to exit.
"""
        instruction.draw()
        win.flip()
        event.waitKeys()
        
        print(f"\nResults for {participant_id}:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  d-prime: {dprime:.2f}")
        print(f"  Exclude: {exclude}")
        
    finally:
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
