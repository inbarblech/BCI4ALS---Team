"""Example program to demonstrate how to send string-valued markers into LSL."""

import random
import time
import os
import imageio as iio


from pylsl import StreamInfo, StreamOutlet


def main():
    #Initialization according to the example 
    print('Please enter subject ID/Name:')
    ID = input()
    print('Hello, ' + ID)
    
    rootFolder = 'C:\\Users\\marko\\bci\\exercises\\Recordings'
    img_folder = "C:\\Users\\marko\\bci\\exercises\\img"
    
    path = os.path.join(rootFolder, 'Sub')
    isExist = os.path.exists(path)
    if(isExist): 
        print(path, " already exists")
    else:
        os.mkdir(path)
    path = os.path.join(path, str(ID))
    isExist = os.path.exists(path)
    if(isExist): 
        print(path, " already exists")
    else:
        os.mkdir(path)
    #Define times
    InitWait = 5;                           # before trials prep time
    trialLength = 5;                        # each trial length in seconds 
    cueLength = 1;                          # time for each cue
    readyLength = 1;                        # time "ready" on screen
    nextLength = 1;                         # time "next" on screen
    
    #Define length and classes
    numTrials = 10;                         # set number of training trials per class (the more classes, the more trials per class)
    numClasses = 3;                         # set number of possible classes
    
    # Set markers / triggers names
    startRecordings = 000;          
    startTrial = 1111;
    Baseline = 1001;
    Idle = 3;
    Left = 1;
    Right = 2;
    endTrial = 9;
    endRecrding = 99;
    
    
    # first create a new stream info (here we set the name to MyMarkerStream,
    # the content-type to Markers, 1 channel, irregular sampling rate,
    # and string-valued data) The last value would be the locally unique
    # identifier for the stream as far as available, e.g.
    # program-scriptname-subjectnumber (you could also omit it but interrupted
    # connections wouldn't auto-recover). The important part is that the
    # content-type is set to 'Markers', because then other programs will know how
    #  to interpret the content
    
    print("Open Lab Recorder & check for MarkerStream and EEG stream, start recording, return here and press enter to continue.")
    i = input()

    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    #info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuniquesourceid23443')

    # next make an outlet
    outlet = StreamOutlet(info) 
    
    trainingImage = []
    trainingImage.append(iio.imread(os.path.join(img_folder, 'square.jpg')))
    trainingImage.append(iio.imread(os.path.join(img_folder, 'arrow_l.jpg')))
    trainingImage.append(iio.imread(os.path.join(img_folder, 'arrow_r.jpg')))


    print("now sending markers...")
    markernames = ['Test', 'Blah', 'Marker', 'XXX', 'Testtest', 'Test-1-2-3']
    while True:
        # pick a sample to send an wait for a bit
        outlet.push_sample([random.choice(markernames)])
        time.sleep(random.random() * 3)


if __name__ == '__main__':
    main()
