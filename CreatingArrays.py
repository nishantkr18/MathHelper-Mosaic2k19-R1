#Created By: nishantKr18
##################################
#     #  #            #   ##      
# #   #  #            #  #  #     
#  #  #  # ##  # ##   #   ##      
#   # #  ##    ##     #  #  #     
#    ##  # ##  #      #   ##      
################################## 

import cv2
import numpy as np
from tqdm import tqdm
import os
import csv

TRAIN_DIR = 'extracted_images/'

labelz = dict(enumerate(['0', '1','2','3','4','5','6','7','8','9','-','+','times','(',')','cos','d','e','infty','int','lim','log','phi','pi','rightarrow','sin','sqrt','tan','y','z']))

training_data =[]
def create_train_data(N):
	count = 0
	for img in tqdm(os.listdir(TRAIN_DIR+labelz[N])):
		path = os.path.join(TRAIN_DIR+labelz[N], img)
		img_data = cv2.imread(path)
		img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
		training_data.append([np.array(img_data), str(N)])
		count+=1
		if(count>4545):
			break

for i in range(len(labelz)):
	create_train_data(i)


print('Shuffling')
np.random.shuffle(training_data)
print('Done Shuffling!!!')

np.save('training_data.npy', training_data)
