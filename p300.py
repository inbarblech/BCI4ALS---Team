import random
import os
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk



STIM_ONSET = 3000 #ms
TIME_BETWEEN_STIMULUS = 2000 #ms
TARGET_NUMBER = 2
NUMBER_OF_BLOCKS = 2
TRIALS_NUMBER = 1
TARGET_RATIO = 0.5
count = 0
#rootFolder = 'C:\\Users\\marko\\bci\\exercises\\Recordings'
img_folder = "C:\\Users\\talyma\\bci\\BCI4ALS---Team\\images"
BLANK = 2
CIRCLE = 1
RECT = 0


def run_training(window, panel, count, trainingImage, training_set, StimOnset, interTime, outlet):
    print (count)
    if(count >= training_set.shape[0]):
        window.destroy()
        return 
    panel.pack_forget()
    if (training_set[count] == CIRCLE): 
        path = trainingImage[CIRCLE]
        marker = CIRCLE
        interval = StimOnset
    else:
        if(training_set[count] == RECT):
            path = trainingImage[RECT]
            marker = RECT
            interval = StimOnset
        else:
            path = trainingImage[BLANK]
            marker = BLANK
            interval = interTime

    markernames = ['Rect', 'Circle', 'inter']
    print("send marker", markernames[marker])

    outlet.push_sample([markernames[marker]])
    
    #window.title(message)
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")    
    #The Pack geometry manager packs widgets in rows or columns.
    window.after(interval, lambda : run_training(window, panel,count+1, trainingImage, training_set, StimOnset, interTime, outlet))
    
    #Start the GUI
    window.mainloop()      

def create_training_set(blocks_N, trials_N, target_ratio):
    tr_len = blocks_N*blocks_N
    training_set = np.zeros(tr_len * 2)
    circle_indexes = random.sample(range(0, tr_len,2), int(target_ratio*tr_len))
    training_set[circle_indexes] = CIRCLE
    training_set[range(1, tr_len *2,2)] = BLANK
    return training_set
    
def main():
    
    # make an outlet
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    outlet = StreamOutlet(info) 
    print("Press enter to start")
    x = input()
    
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
    trainingImage.append(os.path.join(img_folder, 'rect.jpg'))
    trainingImage.append(os.path.join(img_folder, 'circles.jpg'))
    trainingImage.append(os.path.join(img_folder, 'blank.jpg'))
    
   
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
    #create array with entry per test: circle, rectangle or blank
    training_set = create_training_set(blocks_N, trials_N, target_ratio)
    print(training_set)

    #show the relevant trigger and send the relevant marker for each entry in training_set
    
    run_training(window, panel, 0, trainingImage, training_set, StimOnset, interTime, outlet)
            
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
