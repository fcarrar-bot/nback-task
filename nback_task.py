#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual Letter N-Back Task
=========================
A PsychoPy implementation for assessing working memory.

Conditions:
- 0-back: Respond when target letter (X) appears
- 2-back: Respond when current letter matches 2 trials back

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

CONFIG = {
    # Timing (in seconds)
    'stimulus_duration': 0.5,      # 500ms stimulus presentation
    'isi_duration': 2.0,           # 2000ms inter-stimulus interval
    'feedback_duration': 0.5,      # 500ms feedback after response
    
    # Task structure
    'n_levels': [0, 2],            # 0-back and 2-back conditions
    'trials_per_block': 48,        # Trials per block
    'blocks_per_level': 2,         # 2 blocks per N level = 96 trials per level
    'target_ratio': 0.30,          # 30% targets
    
    # Stimuli
    'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
    'target_letter_0back': 'X',    # Target for 0-back condition
    
    # Response keys
    'response_key': 'space',       # Key to press for target
    'quit_key': 'escape',          # Key to quit experiment
    
    # Display
    'letter_size': 0.15,           # Letter size (height units)
    'letter_color': 'white',
    'background_color': 'black',
    
    # Practice
    'practice_trials': 10,         # Practice trials per condition
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sequence(n_trials, n_back, target_ratio, letters, target_letter_0back=None):
    """
    Generate a sequence of letters with specified target ratio.
    
    For 0-back: targets are the specified target letter (X)
    For n-back: targets are letters matching n positions back
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
        
        # Remaining items: some are targets (match n-back), some are non-targets
        remaining = n_trials - n_back
        target_positions = random.sample(range(n_back, n_trials), min(n_targets, remaining))
        
        for i in range(n_back, n_trials):
            if i in target_positions:
                # Target: same as n positions back
                sequence.append(sequence[i - n_back])
                is_target.append(True)
            else:
                # Non-target: different from n positions back
                available = [l for l in letters if l != sequence[i - n_back]]
                # Also avoid creating accidental lures (n-1 back matches)
                if n_back > 1 and len(sequence) >= n_back - 1:
                    available = [l for l in available if l != sequence[i - n_back + 1]]
                sequence.append(random.choice(available))
                is_target.append(False)
    
    return sequence, is_target


def calculate_dprime(hits, misses, false_alarms, correct_rejections):
    """
    Calculate d-prime (sensitivity index).
    
    d' = z(hit_rate) - z(false_alarm_rate)
    
    Applies correction for extreme values (0 or 1).
    """
    from scipy import stats
    
    # Calculate rates
    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    
    if n_targets == 0 or n_nontargets == 0:
        return np.nan
    
    hit_rate = hits / n_targets
    fa_rate = false_alarms / n_nontargets
    
    # Apply Hautus (1995) log-linear correction for extreme values
    # Adjusted rates: (hits + 0.5) / (n_targets + 1)
    hit_rate_adj = (hits + 0.5) / (n_targets + 1)
    fa_rate_adj = (false_alarms + 0.5) / (n_nontargets + 1)
    
    # Calculate z-scores
    z_hit = stats.norm.ppf(hit_rate_adj)
    z_fa = stats.norm.ppf(fa_rate_adj)
    
    dprime = z_hit - z_fa
    return dprime


def calculate_criterion(hits, misses, false_alarms, correct_rejections):
    """
    Calculate response criterion (c).
    
    c = -0.5 * (z(hit_rate) + z(false_alarm_rate))
    """
    from scipy import stats
    
    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    
    if n_targets == 0 or n_nontargets == 0:
        return np.nan
    
    hit_rate_adj = (hits + 0.5) / (n_targets + 1)
    fa_rate_adj = (false_alarms + 0.5) / (n_nontargets + 1)
    
    z_hit = stats.norm.ppf(hit_rate_adj)
    z_fa = stats.norm.ppf(fa_rate_adj)
    
    criterion = -0.5 * (z_hit + z_fa)
    return criterion


# ============================================================================
# EXPERIMENT CLASS
# ============================================================================

class NBackExperiment:
    def __init__(self, participant_id, session=1):
        self.participant_id = participant_id
        self.session = session
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize window
        self.win = visual.Window(
            fullscr=True,
            color=CONFIG['background_color'],
            units='height'
        )
        
        # Initialize stimuli
        self.letter_stim = visual.TextStim(
            self.win,
            text='',
            height=CONFIG['letter_size'],
            color=CONFIG['letter_color'],
            font='Arial'
        )
        
        self.instruction_stim = visual.TextStim(
            self.win,
            text='',
            height=0.03,
            color='white',
            wrapWidth=0.8
        )
        
        self.fixation = visual.TextStim(
            self.win,
            text='+',
            height=0.05,
            color='white'
        )
        
        # Data storage
        self.all_trials = []
        self.summary_data = {}
        
        # Clock
        self.clock = core.Clock()
    
    def show_instructions(self, n_back, is_practice=False):
        """Display instructions for the current condition."""
        practice_text = " (PRACTICE)" if is_practice else ""
        
        if n_back == 0:
            text = f"""
0-BACK TASK{practice_text}

You will see letters appear one at a time.

Press SPACE whenever you see the letter 'X'.

Do NOT press for any other letter.

Try to respond as quickly and accurately as possible.

Press SPACE to begin.
"""
        else:
            text = f"""
{n_back}-BACK TASK{practice_text}

You will see letters appear one at a time.

Press SPACE when the current letter is the SAME 
as the letter shown {n_back} trials ago.

Example: If you see A...B...A, press SPACE on the second A
(because A appeared 2 trials before).

Try to respond as quickly and accurately as possible.

Press SPACE to begin.
"""
        
        self.instruction_stim.text = text
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys(keyList=[CONFIG['response_key']])
    
    def run_block(self, n_back, block_num, n_trials, is_practice=False):
        """Run a single block of trials."""
        
        # Generate sequence
        sequence, targets = generate_sequence(
            n_trials=n_trials,
            n_back=n_back,
            target_ratio=CONFIG['target_ratio'],
            letters=CONFIG['letters'],
            target_letter_0back=CONFIG['target_letter_0back']
        )
        
        block_data = []
        
        for trial_num, (letter, is_target) in enumerate(zip(sequence, targets)):
            # Check for quit
            if event.getKeys(keyList=[CONFIG['quit_key']]):
                self.save_data()
                core.quit()
            
            # Present stimulus
            self.letter_stim.text = letter
            self.letter_stim.draw()
            self.win.flip()
            
            # Collect response during stimulus + ISI
            self.clock.reset()
            response = None
            rt = None
            
            # Stimulus duration
            while self.clock.getTime() < CONFIG['stimulus_duration']:
                keys = event.getKeys(keyList=[CONFIG['response_key']], timeStamped=self.clock)
                if keys and response is None:
                    response = keys[0][0]
                    rt = keys[0][1]
            
            # ISI (show fixation)
            self.fixation.draw()
            self.win.flip()
            
            isi_start = self.clock.getTime()
            while self.clock.getTime() < isi_start + CONFIG['isi_duration']:
                keys = event.getKeys(keyList=[CONFIG['response_key']], timeStamped=self.clock)
                if keys and response is None:
                    response = keys[0][0]
                    rt = keys[0][1]
            
            # Determine response type
            responded = response is not None
            
            if is_target:
                if responded:
                    response_type = 'hit'
                else:
                    response_type = 'miss'
            else:
                if responded:
                    response_type = 'false_alarm'
                else:
                    response_type = 'correct_rejection'
            
            correct = response_type in ['hit', 'correct_rejection']
            
            # Store trial data
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
        
        # Show "practice complete" message
        self.instruction_stim.text = """
Practice complete!

The real task will now begin.

Press SPACE when ready.
"""
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys(keyList=[CONFIG['response_key']])
        
        # Main blocks
        for block in range(1, CONFIG['blocks_per_level'] + 1):
            self.show_instructions(n_back, is_practice=False)
            self.run_block(n_back, block_num=block, n_trials=CONFIG['trials_per_block'], is_practice=False)
            
            # Break between blocks (except after last)
            if block < CONFIG['blocks_per_level']:
                self.instruction_stim.text = f"""
Block {block} of {CONFIG['blocks_per_level']} complete.

Take a short break if needed.

Press SPACE to continue.
"""
                self.instruction_stim.draw()
                self.win.flip()
                event.waitKeys(keyList=[CONFIG['response_key']])
    
    def calculate_summary(self):
        """Calculate summary statistics for each condition."""
        import pandas as pd
        
        df = pd.DataFrame(self.all_trials)
        
        for n_back in CONFIG['n_levels']:
            condition_data = df[(df['n_back'] == n_back) & (df['is_practice'] == False)]
            
            if len(condition_data) == 0:
                continue
            
            hits = len(condition_data[condition_data['response_type'] == 'hit'])
            misses = len(condition_data[condition_data['response_type'] == 'miss'])
            false_alarms = len(condition_data[condition_data['response_type'] == 'false_alarm'])
            correct_rejections = len(condition_data[condition_data['response_type'] == 'correct_rejection'])
            
            total = len(condition_data)
            n_targets = hits + misses
            n_nontargets = false_alarms + correct_rejections
            
            accuracy = (hits + correct_rejections) / total if total > 0 else np.nan
            hit_rate = hits / n_targets if n_targets > 0 else np.nan
            fa_rate = false_alarms / n_nontargets if n_nontargets > 0 else np.nan
            
            dprime = calculate_dprime(hits, misses, false_alarms, correct_rejections)
            criterion = calculate_criterion(hits, misses, false_alarms, correct_rejections)
            
            # RT for hits only
            hit_rts = condition_data[condition_data['response_type'] == 'hit']['rt'].dropna()
            mean_rt = hit_rts.mean() if len(hit_rts) > 0 else np.nan
            
            self.summary_data[f'{n_back}-back'] = {
                'participant_id': self.participant_id,
                'session': self.session,
                'n_back': n_back,
                'total_trials': total,
                'n_targets': n_targets,
                'n_nontargets': n_nontargets,
                'hits': hits,
                'misses': misses,
                'false_alarms': false_alarms,
                'correct_rejections': correct_rejections,
                'accuracy': accuracy,
                'hit_rate': hit_rate,
                'false_alarm_rate': fa_rate,
                'dprime': dprime,
                'criterion': criterion,
                'mean_rt_hits': mean_rt
            }
    
    def save_data(self):
        """Save trial-by-trial and summary data to CSV."""
        import pandas as pd
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Trial-by-trial data
        trial_file = os.path.join(
            self.data_dir, 
            f'{self.participant_id}_session{self.session}_trials_{timestamp}.csv'
        )
        pd.DataFrame(self.all_trials).to_csv(trial_file, index=False)
        print(f"Trial data saved: {trial_file}")
        
        # Summary data
        self.calculate_summary()
        summary_file = os.path.join(
            self.data_dir,
            f'{self.participant_id}_session{self.session}_summary_{timestamp}.csv'
        )
        pd.DataFrame(self.summary_data).T.to_csv(summary_file, index=False)
        print(f"Summary data saved: {summary_file}")
        
        # Append to aggregated file
        agg_file = os.path.join(self.data_dir, 'all_participants_summary.csv')
        summary_df = pd.DataFrame(self.summary_data).T
        
        if os.path.exists(agg_file):
            existing = pd.read_csv(agg_file)
            combined = pd.concat([existing, summary_df], ignore_index=True)
            combined.to_csv(agg_file, index=False)
        else:
            summary_df.to_csv(agg_file, index=False)
        
        print(f"Aggregated data updated: {agg_file}")
        
        return summary_file
    
    def show_end_screen(self):
        """Display end of experiment message."""
        self.instruction_stim.text = """
Thank you for participating!

The experiment is now complete.

Press any key to exit.
"""
        self.instruction_stim.draw()
        self.win.flip()
        event.waitKeys()
    
    def run(self):
        """Run the complete experiment."""
        try:
            # Welcome screen
            self.instruction_stim.text = """
Welcome to the N-Back Working Memory Task

In this experiment, you will see letters appear one at a time.
Your task is to detect specific patterns depending on the condition.

You will complete two types of tasks:
- 0-back: Press SPACE when you see the letter 'X'
- 2-back: Press SPACE when the current letter matches 
          the one from 2 trials ago

Each task includes a short practice before the main trials.

Press SPACE to begin.
"""
            self.instruction_stim.draw()
            self.win.flip()
            event.waitKeys(keyList=[CONFIG['response_key']])
            
            # Run each condition
            for n_back in CONFIG['n_levels']:
                self.run_condition(n_back)
            
            # Save data
            self.save_data()
            
            # End screen
            self.show_end_screen()
            
        finally:
            self.win.close()
            core.quit()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    # Participant info dialog
    dlg = gui.Dlg(title='N-Back Task')
    dlg.addField('Participant ID:', '')
    dlg.addField('Session:', 1)
    
    data = dlg.show()
    
    if not dlg.OK:
        core.quit()
    
    participant_id = data[0]
    session = int(data[1])
    
    if not participant_id:
        print("Error: Participant ID is required")
        core.quit()
    
    # Run experiment
    exp = NBackExperiment(participant_id=participant_id, session=session)
    exp.run()


if __name__ == '__main__':
    main()
