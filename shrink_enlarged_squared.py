#RUN THIS CODE AFTER CHOOSING IMAGE DIMENSIONS AND PATHS BELOW TO GENERATE RESIZED PHOTOS
import cv2
from pathlib import Path
import os
import random

#ADJUST PATHS ACCORDING TO YOUR LOCAL PATHS
directory = "C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/enlarged_squared"
path = 'C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/50x50'


folders_path = []
folders_names = []
x = 1
for fname in os.listdir(directory):

    folder_path = f"{directory}/{fname}"
    folder_path2 = f"{path}/{fname}"
    folder_path3 = f"{path}_test/{fname}"
    folders_path.append(folder_path)
    folders_names.append(fname)
    os.makedirs(folder_path2)
    os.makedirs(folder_path3)


i= 1
x = 0
j = 0

i = 1
j = 0
x = 0
for fname in folders_path:
    rand_list=[]
    while len(rand_list)!=2: # EDIT THIS NUMBER TO ADJUST NUMBER OF IMAGES PER CLASS IN TESTING SET
        x = random.randint(1,16)
        if x not in rand_list:
            rand_list.append(x)
    print(rand_list)
    i = 1
    for img in os.listdir(fname):    
        image = cv2.imread(f"{fname}/{img}",0)
        #EDIT IMAGE SIZE HERE
        resized = cv2.resize(image, (50,50))
        if i in rand_list:
            print("True")
            cv2.imwrite(f"{path}_test/{folders_names[j]}/{folders_names[j]}_{i}.jpg", resized)
        else:
            cv2.imwrite(f"{path}/{folders_names[j]}/{folders_names[j]}_{i}.jpg", resized)
        i+=1
    j+=1
















