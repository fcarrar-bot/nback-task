#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual Letter N-Back Task
=========================
A PsychoPy implementation for assessing working memory.

Conditions:
- 0-back: Respond when target letter (X) appears
- 2-back: Respond when current letter matches 2 trials back
- 3-back: Respond when current letter matches 3 trials back (optional)

Output: CSV with trial-by-trial data + summary metrics including d-prime

Author: Dave (OpenClaw)
Date: 2026-02-07
"""

from psychopy import visual, core, event, data, gui
import os
import random
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Quick demo mode - set to True for short version (~5 min)
DEMO_MODE = True

CONFIG_FULL = {
    # Timing (in seconds)
    'stimulus_duration': 0.5,
    'isi_duration': 2.0,
    
    # Task structure
    'n_levels': [0, 2, 3],          # 0-back, 2-back, and 3-back
    'trials_per_block': 48,
    'blocks_per_level': 2,
    'target_ratio': 0.30,
    
    # Practice
    'practice_trials': 10,
}

CONFIG_DEMO = {
    # Timing (in seconds) - slightly faster for demo
    'stimulus_duration': 0.5,
    'isi_duration': 1.5,
    
    # Task structure - shorter
    'n_levels': [2, 3],              # Skip 0-back in demo, run 2 and 3
    'trials_per_block': 20,          # Fewer trials
    'blocks_per_level': 1,           # Single block
    'target_ratio': 0.30,
    
    # Practice
    'practice_trials': 5,            # Shorter practice
}

# Select config based on mode
CONFIG = CONFIG_DEMO if DEMO_MODE else CONFIG_FULL

# Shared config
CONFIG.update({
    # Stimuli
    'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
    'target_letter_0back': 'X',
    
    # Response keys
    'response_key': 'space',
    'quit_key': 'escape',
    
    # Display
    'letter_size': 0.15,
    'letter_color': 'white',
    'background_color': 'black',
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sequence(n_trials, n_back, target_ratio, letters, target_letter_0back=None):
    """
    Generate a sequence of letters with specified target ratio.
    """
    sequence = []
    is_target = []
    n_targets = int(n_trials * target_ratio)
    
    if n_back == 0:
        # 0-back: randomly place target letters
        target_positions = random.sample(range(n_trials), n_targets)
        for i in range(n_trials):
            if i in target_positions:
                sequence.append(target_letter_0back)
                is_target.append(True)
            else:
                letter = random.choice(letters)
                sequence.append(letter)
                is_target.append(False)
    else:
        # N-back: first n items cannot be targets
        for i in range(n_back):
            sequence.append(random.choice(letters))
            is_target.append(False)
        
        # Remaining items
        remaining = n_trials - n_back
        n_targets_actual = min(n_targets, remaining)
        target_positions = random.sample(range(n_back, n_trials), n_targets_actual)
        
        for i in range(n_back, n_trials):
            if i in target_positions:
                sequence.append(sequence[i - n_back])
                is_target.append(True)
            else:
                available = [l for l in letters if l != sequence[i - n_back]]
                if n_back > 1 and len(sequence) >= n_back - 1:
                    available = [l for l in available if l != sequence[i - n_back + 1]]
                sequence.append(random.choice(available) if available else random.choice(letters))
                is_target.append(False)
    
    return sequence, is_target


def calculate_dprime(hits, misses, false_alarms, correct_rejections):
    """Calculate d-prime (sensitivity index) with log-linear correction."""
    from scipy import stats
    
    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    
    if n_targets == 0 or n_nontargets == 0:
        return np.nan
    
    hit_rate_adj = (hits + 0.5) / (n_targets + 1)
    fa_rate_adj = (false_alarms + 0.5) / (n_nontargets + 1)
    
    z_hit = stats.norm.ppf(hit_rate_adj)
    z_fa = stats.norm.ppf(fa_rate_adj)
    
    return z_hit - z_fa


def calculate_criterion(hits, misses, false_alarms, correct_rejections):
    """Calculate response criterion (c)."""
    from scipy import stats
    
    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    
    if n_targets == 0 or n_nontargets == 0:
        return np.nan
    
    hit_rate_adj = (hits + 0.5) / (n_targets + 1)
    fa_rate_adj = (false_alarms + 0.5) / (n_nontargets + 1)
    
    z_hit = stats.norm.ppf(hit_rate_adj)
    z_fa = stats.norm.ppf(fa_rate_adj)
    
    return -0.5 * (z_hit + z_fa)


# ============================================================================
# EXPERIMENT CLASS
# ============================================================================

class NBackExperiment:
    def __init__(self, participant_id, session=1):
        self.participant_id = participant_id
        self.session = session
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.win = visual.Window(
            fullscr=True,
            color=CONFIG['background_color'],
            units='height'
        )
        
        self.letter_stim = visual.TextStim(
            self.win, text='', height=CONFIG['letter_size'],
            color=CONFIG['letter_color'], font='Arial'
        )
        
        self.instruction_stim = visual.TextStim(
            self.win, text='', height=0.03, color='white', wrapWidth=0.8
        )
        
        self.fixation = visual.TextStim(
            self.win, text='+', height=0.05, color='white'
        )
        
        self.all_trials = []
        self.summary_data = {}
        self.clock = core.Clock()
    
    def show_instructions(self, n_back, is_practice=False):
        """Display instructions for the current condition."""
        practice_text = " (PRACTICE)" if is_practice else ""
        mode_text = "[DEMO MODE - Short Version]\n\n" if DEMO_MODE else ""
        
        if n_back == 0:
            text = f"""{mode_text}0-BACK TASK{practice_text}

Press SPACE whenever you see the letter 'X'.

Press SPACE to begin."""
        else:
            text = f"""{mode_text}{n_back}-BACK TASK{practice_text}

Press SPACE when the current letter is the SAME 
as the letter shown {n_back} trials ago.

{"Example: A...B...C...A → press on 2nd A (matches 3 back)" if n_back == 3 else "Example: A...B...A → press on 2nd A (matches 2 back)"}

Press SPACE to begin."""
        
        self.instruction_stim.text = text
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys(keyList=[CONFIG['response_key']])
    
    def run_block(self, n_back, block_num, n_trials, is_practice=False):
        """Run a single block of trials."""
        
        sequence, targets = generate_sequence(
            n_trials=n_trials,
            n_back=n_back,
            target_ratio=CONFIG['target_ratio'],
            letters=CONFIG['letters'],
            target_letter_0back=CONFIG['target_letter_0back']
        )
        
        block_data = []
        
        for trial_num, (letter, is_target) in enumerate(zip(sequence, targets)):
            if event.getKeys(keyList=[CONFIG['quit_key']]):
                self.save_data()
                core.quit()
            
            # Present stimulus
            self.letter_stim.text = letter
            self.letter_stim.draw()
            self.win.flip()
            
            self.clock.reset()
            response = None
            rt = None
            
            # Stimulus duration
            while self.clock.getTime() < CONFIG['stimulus_duration']:
                keys = event.getKeys(keyList=[CONFIG['response_key']], timeStamped=self.clock)
                if keys and response is None:
                    response = keys[0][0]
                    rt = keys[0][1]
            
            # ISI
            self.fixation.draw()
            self.win.flip()
            
            isi_start = self.clock.getTime()
            while self.clock.getTime() < isi_start + CONFIG['isi_duration']:
                keys = event.getKeys(keyList=[CONFIG['response_key']], timeStamped=self.clock)
                if keys and response is None:
                    response = keys[0][0]
                    rt = keys[0][1]
            
            responded = response is not None
            
            if is_target:
                response_type = 'hit' if responded else 'miss'
            else:
                response_type = 'false_alarm' if responded else 'correct_rejection'
            
            correct = response_type in ['hit', 'correct_rejection']
            
            trial_data = {
                'participant_id': self.participant_id,
                'session': self.session,
                'n_back': n_back,
                'block': block_num,
                'trial': trial_num + 1,
                'is_practice': is_practice,
                'letter': letter,
                'is_target': is_target,
                'responded': responded,
                'response_type': response_type,
                'correct': correct,
                'rt': rt if rt else np.nan,
                'timestamp': datetime.now().isoformat()
            }
            
            block_data.append(trial_data)
            
            if not is_practice:
                self.all_trials.append(trial_data)
        
        return block_data
    
    def run_condition(self, n_back):
        """Run all blocks for a given N-back condition."""
        
        # Practice
        self.show_instructions(n_back, is_practice=True)
        self.run_block(n_back, block_num=0, n_trials=CONFIG['practice_trials'], is_practice=True)
        
        self.instruction_stim.text = "Practice complete!\n\nPress SPACE when ready."
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys(keyList=[CONFIG['response_key']])
        
        # Main blocks
        for block in range(1, CONFIG['blocks_per_level'] + 1):
            self.show_instructions(n_back, is_practice=False)
            self.run_block(n_back, block_num=block, n_trials=CONFIG['trials_per_block'], is_practice=False)
            
            if block < CONFIG['blocks_per_level']:
                self.instruction_stim.text = f"Block {block}/{CONFIG['blocks_per_level']} complete.\n\nPress SPACE to continue."
                self.instruction_stim.draw()
                self.win.flip()
                event.waitKeys(keyList=[CONFIG['response_key']])
    
    def calculate_summary(self):
        """Calculate summary statistics for each condition."""
        import pandas as pd
        
        df = pd.DataFrame(self.all_trials)
        
        for n_back in CONFIG['n_levels']:
            cond = df[(df['n_back'] == n_back) & (df['is_practice'] == False)]
            
            if len(cond) == 0:
                continue
            
            hits = len(cond[cond['response_type'] == 'hit'])
            misses = len(cond[cond['response_type'] == 'miss'])
            fas = len(cond[cond['response_type'] == 'false_alarm'])
            crs = len(cond[cond['response_type'] == 'correct_rejection'])
            
            total = len(cond)
            n_targets = hits + misses
            n_nontargets = fas + crs
            
            accuracy = (hits + crs) / total if total > 0 else np.nan
            hit_rate = hits / n_targets if n_targets > 0 else np.nan
            fa_rate = fas / n_nontargets if n_nontargets > 0 else np.nan
            
            dprime = calculate_dprime(hits, misses, fas, crs)
            criterion = calculate_criterion(hits, misses, fas, crs)
            
            hit_rts = cond[cond['response_type'] == 'hit']['rt'].dropna()
            mean_rt = hit_rts.mean() if len(hit_rts) > 0 else np.nan
            
            self.summary_data[f'{n_back}-back'] = {
                'participant_id': self.participant_id,
                'session': self.session,
                'n_back': n_back,
                'total_trials': total,
                'hits': hits,
                'misses': misses,
                'false_alarms': fas,
                'correct_rejections': crs,
                'accuracy': round(accuracy, 3),
                'hit_rate': round(hit_rate, 3) if not np.isnan(hit_rate) else np.nan,
                'false_alarm_rate': round(fa_rate, 3) if not np.isnan(fa_rate) else np.nan,
                'dprime': round(dprime, 2) if not np.isnan(dprime) else np.nan,
                'criterion': round(criterion, 2) if not np.isnan(criterion) else np.nan,
                'mean_rt_hits': round(mean_rt * 1000, 0) if not np.isnan(mean_rt) else np.nan  # in ms
            }
    
    def save_data(self):
        """Save trial-by-trial and summary data to CSV."""
        import pandas as pd
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Trial data
        trial_file = os.path.join(self.data_dir, f'{self.participant_id}_s{self.session}_trials_{timestamp}.csv')
        pd.DataFrame(self.all_trials).to_csv(trial_file, index=False)
        
        # Summary
        self.calculate_summary()
        summary_file = os.path.join(self.data_dir, f'{self.participant_id}_s{self.session}_summary_{timestamp}.csv')
        summary_df = pd.DataFrame(self.summary_data).T
        summary_df.to_csv(summary_file, index=False)
        
        # Aggregated
        agg_file = os.path.join(self.data_dir, 'all_participants_summary.csv')
        if os.path.exists(agg_file):
            existing = pd.read_csv(agg_file)
            combined = pd.concat([existing, summary_df], ignore_index=True)
            combined.to_csv(agg_file, index=False)
        else:
            summary_df.to_csv(agg_file, index=False)
        
        print(f"\nData saved to: {self.data_dir}")
        print(f"  Trials: {trial_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Aggregated: {agg_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        for condition, metrics in self.summary_data.items():
            print(f"\n{condition.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"  d-prime:  {metrics['dprime']}")
            print(f"  Hits: {metrics['hits']}, Misses: {metrics['misses']}, FA: {metrics['false_alarms']}")
            if not np.isnan(metrics['mean_rt_hits']):
                print(f"  Mean RT:  {metrics['mean_rt_hits']:.0f} ms")
        print("="*50)
        
        return summary_file
    
    def show_end_screen(self):
        """Display end of experiment message with results."""
        results_text = "RESULTS:\n\n"
        for condition, metrics in self.summary_data.items():
            results_text += f"{condition}: {metrics['accuracy']*100:.0f}% accuracy, d'={metrics['dprime']}\n"
        
        self.instruction_stim.text = f"""
Experiment Complete!

{results_text}
Thank you for participating.

Press any key to exit.
"""
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys()
    
    def run(self):
        """Run the complete experiment."""
        try:
            mode_note = "DEMO MODE (~5 min)\n\n" if DEMO_MODE else ""
            levels_desc = ", ".join([f"{n}-back" for n in CONFIG['n_levels']])
            
            self.instruction_stim.text = f"""{mode_note}Welcome to the N-Back Working Memory Task

You will complete: {levels_desc}

Your task: Press SPACE when you detect a target.
Each condition will be explained before it starts.

Press SPACE to begin.
"""
            self.instruction_stim.draw()
            self.win.flip()
            event.waitKeys(keyList=[CONFIG['response_key']])
            
            for n_back in CONFIG['n_levels']:
                self.run_condition(n_back)
            
            self.save_data()
            self.show_end_screen()
            
        finally:
            self.win.close()
            core.quit()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    dlg = gui.Dlg(title='N-Back Task')
    dlg.addField('Participant ID:', 'test')
    dlg.addField('Session:', 1)
    dlg.addField('Demo mode (short):', DEMO_MODE)
    
    data = dlg.show()
    
    if not dlg.OK:
        core.quit()
    
    participant_id = data[0]
    session = int(data[1])
    
    # Update demo mode based on dialog
    global DEMO_MODE, CONFIG
    DEMO_MODE = data[2]
    CONFIG = CONFIG_DEMO if DEMO_MODE else CONFIG_FULL
    CONFIG.update({
        'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
        'target_letter_0back': 'X',
        'response_key': 'space',
        'quit_key': 'escape',
        'letter_size': 0.15,
        'letter_color': 'white',
        'background_color': 'black',
    })
    
    if not participant_id:
        print("Error: Participant ID required")
        core.quit()
    
    exp = NBackExperiment(participant_id=participant_id, session=session)
    exp.run()


if __name__ == '__main__':
    main()
