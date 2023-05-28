from pylsl import StreamInfo, StreamOutlet
import os
import numpy as np
import random
import tkinter as tk
from PIL import Image, ImageTk
import pyxdf

#general parameters
img_folder = os.path.join(os.getcwd(), "images")
recording_folder = os.path.join(os.getcwd(), "Recordings\\EGI.xdf")

StimOnset = 3000  # ms
interTime = 2000  # ms between stimulus
blocks_N = 6
trials_N = 6
target_ratio = 0.2
count = 0

START = 3
BLANK = 2
TARGET = 1
OTHER = 0

def run_block(window, panel, block_num, trail_num, cur_target, training_set, StimOnset, interTime, outlet,  images_list, target_list):
    print(trail_num)
    if trail_num >= trials_N * 2:  # end of training
        window.destroy()
        return
    panel.pack_forget()

    if training_set[block_num * trials_N + trail_num] == TARGET:
        cur_target_num = random.choice(target_list)
        path = os.path.join(img_folder, cur_target_num)
        marker = 0
        interval = StimOnset
    elif training_set[block_num * trials_N + trail_num] == OTHER:
        index = random.choice([i for i in images_list if i not in target_list and i != 'blank.jpg'])
        path = os.path.join(img_folder, index)
        marker = 1
        interval = StimOnset
    elif training_set[block_num * trials_N + trail_num] == START:
        path = os.path.join(img_folder, 'Count Images', cur_target + '.PNG')
        marker = 2
        interval = interTime
    else:
        path = os.path.join(img_folder, 'blank.jpg')
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
    window.after(interval, lambda: run_block(window, panel, block_num, trail_num + 1, cur_target, training_set, StimOnset, interTime, outlet,  images_list, target_list))


    # Start the GUI
    window.mainloop()


def create_training_set(blocks_N, trials_N, target_ratio):
    tr_len = blocks_N * trials_N
    training_set = np.zeros(tr_len * 2)
    for block in range(blocks_N):
        target_indexes = random.sample(range(0, trials_N * 2, 2), int(target_ratio * trials_N))
        target_indexes = [index + (block * trials_N * 2) for index in target_indexes]
        training_set[target_indexes] = TARGET
    training_set[range(1, tr_len * 2, 2)] = BLANK
    training_set = np.insert(training_set,0,START,axis=0)
    return training_set

def run_training(img_folder, cur_target, training_set, StimOnset, interTime, outlet, blocks_N):
    images_list = os.listdir(img_folder)
    if '.DS_Store' in images_list:
        images_list.remove('.DS_Store')
    images_list.remove('Count Images')
    images_list.remove('old')
    target_list = [i for i in images_list if cur_target in i]
    for block_idx in range(blocks_N):
        """GUI
        Recommended to add GUI to control experiment parameters"""
        # This creates the main window of an application
        # animal_index = random.choice([i for i in range(1, len(animals_list))])

        window = tk.Tk()
        window.geometry("3000x3000")
        window.configure(background='grey')
        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        img = ImageTk.PhotoImage(Image.open(os.path.join(img_folder, 'blank.jpg')))
        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(window, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")
        # animal_index = random.choice([i for i in range(1, len(trainingImage))])
        print("Look for " + cur_target)
        run_block(window, panel, block_idx, 0, cur_target, training_set, StimOnset, interTime, outlet, images_list, target_list)

def main():
    # make an outlet
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    outlet = StreamOutlet(info)
    print('select animal for training: 1=horse, 2=cat, 3=butterfly, 4=bird, 5=bear')
    # print("Press enter to start")
    x = int(input())

    animals_dic = {1:'horse', 2:'cat', 3:'butterfly', 4:'bird', 5:'bear'}
    cur_target = animals_dic[x]

    training_set = create_training_set(blocks_N, trials_N, target_ratio)
    run_training(img_folder, cur_target, training_set, StimOnset, interTime, outlet, blocks_N)

    print("Training done")
    x = input()
    data, header = pyxdf.load_xdf(recording_folder)
    print(data)

if __name__ == '__main__':
    main()