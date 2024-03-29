import random
import os
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from pathlib import Path
import pyxdf

STIM_ONSET = 3000  # ms
TIME_BETWEEN_STIMULUS = 2000  # ms
TARGET_NUMBER = 2
NUMBER_OF_BLOCKS = 6
TRIALS_NUMBER = 6
TARGET_RATIO = 0.2
count = 0
# rootFolder = 'C:\\Users\\marko\\bci\\exercises\\Recordings'
# img_folder = "C:\\Users\\talyma\\bci\\BCI4ALS---Team\\images"
# img_folder = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\images"
# recording_folder = 'C:\\Users\\marko\\bci\\exercises\\Recordings\\EGI.xdf'
img_folder = os.path.join(os.getcwd(), "images")
recording_folder = os.path.join(os.getcwd(), "Recordings\\EGI.xdf")
BLANK = 2
TARGET = 1
OTHER = 0


def run_block(window, panel, block_num, trail_num, trainingImage, training_set, StimOnset, interTime, outlet,
              animal_index):
    print(trail_num)
    if trail_num >= TRIALS_NUMBER * 2:  # end of training
        window.destroy()
        return
    panel.pack_forget()
    if training_set[block_num * TRIALS_NUMBER + trail_num] == TARGET:
        path = trainingImage[animal_index]
        marker = 0
        interval = StimOnset
    else:
        if training_set[block_num * TRIALS_NUMBER + trail_num] == OTHER:
            index = random.choice([i for i in range(1, len(trainingImage)) if i != animal_index])
            path = trainingImage[index]
            marker = 1
            interval = StimOnset
        else:
            path = trainingImage[0]
            marker = 2
            interval = interTime

    markernames = ['Target', 'Other', 'inter']
    print("send marker", markernames[marker])

    outlet.push_sample([markernames[marker]])

    # window.title(message)
    # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = Image.open(path)
    img = img.resize((1500, 1000))
    img = ImageTk.PhotoImage(img)
    # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    # The Pack geometry manager packs widgets in rows or columns.
    window.after(interval,
                 lambda: run_block(window, panel, block_num, trail_num + 1, trainingImage, training_set, StimOnset,
                                   interTime, outlet, animal_index))

    # Start the GUI
    window.mainloop()


def run_training(trainingImage, training_set, StimOnset, interTime, outlet):
    for block_idx in range(NUMBER_OF_BLOCKS):
        """GUI
        Recommended to add GUI to control experiment parameters"""
        # This creates the main window of an application
        window = tk.Tk()
        window.geometry("3000x3000")
        window.configure(background='grey')
        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        img = ImageTk.PhotoImage(Image.open(os.path.join(img_folder, 'blank.jpg')))
        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(window, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")
        animal_index = random.choice([i for i in range(1, len(trainingImage))])
        print(f"Look for {Path(trainingImage[animal_index]).stem}")
        run_block(window, panel, block_idx, 0, trainingImage, training_set, StimOnset, interTime, outlet, animal_index)


def create_training_set(blocks_N, trials_N, target_ratio):
    tr_len = blocks_N * trials_N
    training_set = np.zeros(tr_len * 2)
    for block in range(blocks_N):
        target_indexes = random.sample(range(0, trials_N * 2, 2), int(target_ratio * trials_N))
        target_indexes = [index + (block * trials_N * 2) for index in target_indexes]
        training_set[target_indexes] = TARGET
    training_set[range(1, tr_len * 2, 2)] = BLANK
    return training_set


def main():
    # make an outlet
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    outlet = StreamOutlet(info)
    print("Press enter to start")
    x = input()

    """Experiment parameters"""
    StimOnset = STIM_ONSET
    interTime = TIME_BETWEEN_STIMULUS
    targets_N = TARGET_NUMBER
    # stimulusType = (type of stimulus to load and present- different pictures\ audio \ etc.)
    blocks_N = NUMBER_OF_BLOCKS
    trials_N = TRIALS_NUMBER
    target_ratio = TARGET_RATIO

    # """Images"""
    trainingImage = []
    trainingImage.append(os.path.join(img_folder, 'blank.jpg'))  # index 0 of other is blank
    trainingImage.append(os.path.join(img_folder, 'cat.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'bear.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'butterfly.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'eagle.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'fox.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'monkey.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'squirrel.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'bird.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'bird2.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'bird3.jpeg'))
    trainingImage.append(os.path.join(img_folder, 'bird4.jpeg'))

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
    # taly:
    # create array with entry per test: circle, rectangle or blank
    training_set = create_training_set(blocks_N, trials_N, target_ratio)
    print(training_set)

    # show the relevant trigger and send the relevant marker for each entry in training_set
    run_training(trainingImage, training_set, StimOnset, interTime, outlet)

    print("Training done")
    x = input()
    data, header = pyxdf.load_xdf(recording_folder)
    print(data)

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


if __name__ == '__main__':
    main()