# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:05:43 2023

@author: marko
"""
import pygame
import time
import numpy as np
import random
from pylsl import StreamInfo, StreamOutlet


STIM_ONSET = 500 #ms
TIME_BETWEEN_STIMULUS = 500 #ms
NUMBER_OF_BLOCKS = 1
TRIALS_NUMBER = 300
TARGET_RATIO     = 0.1

BLANCK = 0
TARGET_Y = 1
TARGET_N = 2
GAP_FILLER = 3

LIGHT_ON = 0
LIGHT_OFF = 1
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
debug = True

markernames = ['LIGHT_ON-t', 'LIGHT_ON', 'LIGHT_OFF-t', 'LIGHT_OFF', 'gap filler', 'blank', 'block end', 'all done']


from screeninfo import get_monitors

def update_exclude_indexes(all_indexes, index, exclude_indexes):
    if index not in exclude_indexes: exclude_indexes.append(index)
    if index-2 in all_indexes: 
        if index-2 not in exclude_indexes: 
            exclude_indexes.append(index-2)
    if index+2 in exclude_indexes: 
        if index+2 not in exclude_indexes: 
            exclude_indexes.append(index+2)
    return exclude_indexes

def set_target_indexes(trials_N, number_of_targets):
    all_indexes = range(0,trials_N*2,2)
    target_indexes = random.sample(all_indexes, number_of_targets*2) #randomly pick "y" and "n"
    number_of_indexes = len(target_indexes)
    fixed_target_indexes = []
    exclude_indexes = []
    exclude_indexes = update_exclude_indexes(all_indexes, target_indexes[0], exclude_indexes)
    for index in target_indexes: #make sure that there will be no sequence appearances 
        neigbour = index + 2
        if neigbour in fixed_target_indexes: continue
        neigbour = index - 2
        if neigbour in fixed_target_indexes: continue
        fixed_target_indexes.append(index) 
        exclude_indexes = update_exclude_indexes(all_indexes, index, exclude_indexes)
    missing_indexes = number_of_indexes - len(fixed_target_indexes)
    print(missing_indexes, len(fixed_target_indexes),number_of_indexes)
    choose_from = np.setdiff1d(all_indexes, exclude_indexes)
    add_indexes = random.sample(sorted(choose_from), missing_indexes) 
    fixed_target_indexes.extend(add_indexes)
    print(len(fixed_target_indexes))

    return fixed_target_indexes

def create_training_set(blocks_N:int, trials_N:int, target_ratio:float):
    tr_len = blocks_N*trials_N
    training_set = np.zeros(tr_len*2)
    training_set[range(0, tr_len *2, 2)] = GAP_FILLER
    number_of_targets = int(target_ratio*trials_N)    
    for block in range(blocks_N):
        target_indexes = set_target_indexes(trials_N, number_of_targets)
        target_indexes_y = random.sample(target_indexes, number_of_targets) #randomly choose half to be "yes"
        target_indexes_n = np.setdiff1d(target_indexes, target_indexes_y) #set the rest to "no"

        target_y_indexes = [index + (block*trials_N*2) for index in target_indexes_y]
        target_n_indexes = [index + (block*trials_N*2) for index in target_indexes_n]
        training_set[target_y_indexes] = TARGET_Y
        training_set[target_n_indexes] = TARGET_N
    
    #choose target for each block
    if(blocks_N <=1):
        targets = np.array([LIGHT_OFF])
    else:
        targets = np.zeros(blocks_N)
        triangle_blocks = random.sample(range(0, blocks_N, 1), int(blocks_N/2))
        targets[triangle_blocks] = LIGHT_OFF
    print(targets)
    return training_set, targets
def create_online_set():
    tr_len = 10
    online_set = np.zeros(tr_len*2)
    online_set[range(0, tr_len *2, 2)] = GAP_FILLER
    t_o = random.sample(range(10),2)
    target_y_indexes = [t_o[0]]
    target_n_indexes = [t_o[1]]

    online_set[target_y_indexes] = TARGET_Y
    online_set[target_n_indexes] = TARGET_N
    
    targets = np.array([LIGHT_OFF])
    return online_set, targets

def draw_params(width:int, height:int):
    margine_x = 0.1*width
    margine_y = 0.1*height    
    rect = pygame.Rect(margine_x,margine_y,width- margine_x*2,height-margine_y*2)
    tri = ((margine_x,margine_y), (width - margine_x,margine_y), (width/2,height*0.9))
    return rect, tri

def send_marker(value, outlet):
    outlet.push_sample([value])
    if(value!='gap filler' and value!='blank'): print("send", value)


def present_paradigm(training_set:np.array, target:np.array, width:int, height:int, outlet, conn = None):
    print(width, height)
    pygame.init()
    # Set up the drawing window
    screen = pygame.display.set_mode([width, height]) 
    font = pygame.font.Font("./img/Example.ttf", 100)
    clock = pygame.time.Clock()
    # Run until the user asks to quit or untill done 
    running = True
    rect, tri = draw_params(width, height)
    for block in range(NUMBER_OF_BLOCKS):
        current_set = training_set[block*TRIALS_NUMBER*2:(block+1)*TRIALS_NUMBER*2]
        print("current set", current_set)
        current_target = target[block]
        if(current_target == LIGHT_ON):
            text = font.render("Please turn light on", True, white, blue)
        else:
            text = font.render("Please turn light off", True, white, blue)
        textRect = text.get_rect()
        textRect.center = (width // 2, height // 2)
        screen.fill(white)
        screen.blit(text, textRect)
        pygame.display.update()
        #send "start of block to liblsl"
        clock.tick(0.5)
        print("current_target",current_target)
        for action in current_set:
            if(running == False):return
        
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
            # Fill the background with white
            screen.fill((255, 255, 255))            

            clock_tick = 1000/STIM_ONSET
            marker = 5
            if (action == GAP_FILLER):
                imp = pygame.image.load("./img/gf.png").convert()
                marker = 4
            elif(action == TARGET_Y):
                if(current_target == LIGHT_ON):
                    imp = pygame.image.load("./img/on.png").convert()
                    marker = 0
                else:
                    imp = pygame.image.load("./img/off.png").convert()
                    marker = 2
            elif(action == TARGET_N):                
                if(current_target == LIGHT_OFF):
                    imp = pygame.image.load("./img/on.png").convert()
                    marker = 1
                else:
                    imp = pygame.image.load("./img/off.png").convert()
                    marker = 3
            elif(action == BLANCK):
                imp = pygame.image.load("./img/blank.png").convert()
                clock_tick = 1000/TIME_BETWEEN_STIMULUS
            # Using blit to copy content from one surface to other
            screen.blit(imp, (width/3, height/3))
            start_time = time.time_ns()    
            send_marker(markernames[marker], outlet)            
            if(debug): print("time sending outlet",time.time_ns()-start_time)
            start_time = time.time_ns()
            # Flip the display
            pygame.display.flip()
            if(debug): print("time presenting flip",time.time_ns()-start_time)
                        
            clock.tick(clock_tick)
            

        send_marker(markernames[6], outlet) #end of block
        if(conn !=None): 
            print("Wait for processing")
            conn.recv()
    send_marker(markernames[7], outlet) #all done 
 
    # Done! Time to quit.
    pygame.quit()

def set_outlet():
    # make an outlet
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
    outlet = StreamOutlet(info) 
    print("Press enter to start") 
    x = input()
    return outlet

def get_screen_param():
    for m in get_monitors():
        first_monitor = m
        print(str(m))
        break
    width = int(first_monitor.width *0.95)
    height = int(first_monitor.height*0.90)
    return width,height


    
if __name__ == '__main__':
    width,height = get_screen_param()
    training_set, targets = create_training_set(blocks_N = NUMBER_OF_BLOCKS, trials_N = TRIALS_NUMBER, target_ratio = TARGET_RATIO)
    print(len(training_set[training_set==1]), len(training_set[training_set==2]), len(training_set[training_set==3]))
    print(targets)
    outlet = set_outlet() 
 
    present_paradigm(training_set, targets, width, height, outlet)
    
  