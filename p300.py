import random
import time
import os
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import *
from PIL import Image, ImageTk
import imageio as iio



STIM_ONSET = 1
TIME_BETWEEN_STIMULUS = 1
TARGET_NUMBER = 2
NUMBER_OF_BLOCKS = 5
TRIALS_NUMBER = 10
TARGET_RATIO = 0.5
count = 0
#rootFolder = 'C:\\Users\\marko\\bci\\exercises\\Recordings'
img_folder = "C:\\Users\\talyma\\bci\\BCI4ALS---Team\\images"


def run_training(window, panel, count):
    print (count)
    if(count > 1):
        window.destroy()
        return 
    panel.pack_forget()
    if (count == 0): 
        path = os.path.join(img_folder, 'circles.jpg')
        message = "Please count circles"
    else:
        message = "Please focus on rectangle"
        path = os.path.join(img_folder, 'rect.jpg')
    window.title(message)
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")    
    #The Pack geometry manager packs widgets in rows or columns.
    window.after(2000, lambda : run_training(window, panel,count+1))
    #Start the GUI
    window.mainloop()      

    
def main():
    
    
    """GUI
    Recommended to add GUI to control experiment parameters"""
    #This creates the main window of an application
    window = tk.Tk()
    window.geometry("3000x3000")
    window.configure(background='grey')
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(os.path.join(img_folder, 'blank.jpg')))
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")    

    
    """Experiment parameters"""
    StimOnset = STIM_ONSET
    interTime = TIME_BETWEEN_STIMULUS
    targets_N = TARGET_NUMBER
    #stimulusType = (type of stimulus to load and present- different pictures\ audio \ etc.)
    blocks_N = NUMBER_OF_BLOCKS
    trials_N = TRIALS_NUMBER
    target_ratio = TARGET_RATIO
    
   # """Images"""
    trainingImage = []
    trainingImage.append(iio.imread(os.path.join(img_folder, 'circles.jpg')))
    trainingImage.append(iio.imread(os.path.join(img_folder, 'rect.jpg')))
    nontarget = iio.imread(os.path.join(img_folder, 'blank.jpg'))
    
   
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
    #taly: 
    run_training(window, panel, 0)
        
    
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
