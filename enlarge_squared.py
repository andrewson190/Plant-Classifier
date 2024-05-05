
#CODE TO ORIGINALLY ENLARGE PHOTOS TO SQUARE WITH FILLED BLACK SPACE
#DO NOT RUN AGAIN, ALREADY GENERATED WITH enlarged_squared FOLDER

import cv2
from pathlib import Path
import os

directory = "C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/images"
path = 'C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/images_enlarged'


folders_path = []
folders_names = []
x = 1
for fname in os.listdir(directory):

    folder_path = f"{directory}/{fname}"
    folder_path2 = f"{path}/{fname}"
    folders_path.append(folder_path)
    folders_names.append(fname)
    os.makedirs(folder_path2)

i = 1
j = 0
x = 0
for fname in folders_path:
    for img in os.listdir(fname):
        image = cv2.imread(f"{fname}/{img}",0)
        h, w = image.shape
        if h<=w: 
            x = w-h
            larger_img = cv2.copyMakeBorder(image, 0, x, 0, 0, cv2.BORDER_CONSTANT, 0)
        else:
            x = h-w
            larger_img = cv2.copyMakeBorder(image, 0, 0, 0, x, cv2.BORDER_CONSTANT, 0)

        cv2.imwrite(os.path.join(f"{path}/{folders_names[j]}" , f"{i}.jpg"), larger_img)
        i+=1
    j+=1











