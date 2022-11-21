Parameters

"""GUI
Recommended to add GUI to control experiment parameters"""

"""Experiment parameters"""
StimOnset = (time to present the stimulus)
interTime = (time between stimulus)
targets_N = (number of target stimulus)
stimulusType = (type of stimulus to load and present- different pictures\ audio \ etc.)
blocks_N = (number of blocks)
trials_N = (number of trials per block)
target_ratio = (percentage of the oddball onsets)

"""Images"""
Target_i = load(img_1)  " Load images (or any other stimulus type per target)"
nontarget = load(img_nontarget)

"""EEG Params"""
"""Differ between systems, but the parameters needed to stream data: Using brainflow will cover most of
this part. (All you need is to set the port)"""

"Run training experiment"
"""For number of blocks:
Display the current target of the block (e.g., 'Count number of circles')
present number of block(can wait for key press for readiness)
Create training vector of target and non - target sequence write trigger('block beginning')
For number of trials:
Display stimulus (according to training vector)
write trigger(corresponding to current stimulus)
wait(StimOnset)
Stop stimulus onset
Wait(interTime)
Output should be:
raw EEG with triggers marking the stimuli onset (per type) and the blocks beginnings."""

"Preprocessing: Bandpass filter."

"Segment & average the data"
"""Choose desired ERP segment(i.e. - 200 ms: 500 ms)
Cut each stimuli type per block according to segment limit
Subtract baseline(subtract mean amplitude of pre stimuli from the entire trial)
Average the trials(create ERP per class per block)
Output should be:
    per class(number of target + nontarget) ERP per block per electrode
electrodes x ERP_time x blocks_N(per class)"""

"Feature extraction"
"""Choose features to extract (i.e., down sample the signal from chosen electrodes)
extract per block the features from the ERP
Create labeled feature matrix
Labels should be binary (target or non-target)
according to the current target of the specific block
(even if there is more than one oddball, each block should have 1 target to focus on)"""

"Model training"
"""Choose a model to use
Fit the model with the features and labels
Evaluate the model (using CV or test set)"""

"Online evaluation"
"""Run same paradigm as in training
after each block pass the data from the block with segments 3 and 4 (segmentation and feature extraction).
Use the pre-trained model to classify the ERP to P300 and non-P300 components.
Display the target type corresponding to the P300 ERP."""
