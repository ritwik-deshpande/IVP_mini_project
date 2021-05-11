import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
import cv2
import sys
import csv
from keras.models import load_model

class DisjointSet(object):

    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])


def number_recognition(img,model):
    

    plt.imshow(img,cmap='Greys')

    dim=(28,28)
    crop_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    plt.imshow(crop_img,cmap='Greys')

    im2arr = np.array(crop_img)
    im2arr = im2arr.reshape(1,28,28,1)

    #plt.title('Sample Image')
    #plt.imshow(img,cmap='Greys')
    #plt.show()


    #plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    #pred = modelcnn1.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
    pred = model.predict(im2arr)
    return pred.argmax()


def connectedComponentLabelling(img):
    img2 = np.zeros(img.shape[:2],dtype='int32')
    
    label = 1
    label_coordinates = {}
    length,width = img.shape[:2]
    equivalent_list = []
    t_label = int()
    r = int()
    t = int()
    for i in range(0,length):
        for j in range(0,width):
            if img[i][j] == 1:
                if i == 0:
                    r = 0
                else:
                    r = img[i-1][j]
                if j == 0:
                    t = 0
                else:
                    t = img[i][j-1]
                    
                if r==0 and t==0:
                    img2[i][j] = label
                    label = label + 1
                elif r== 0 and t == 1:
                    img2[i][j] = img2[i][j-1]
                elif r== 1 and t == 0:
                    img2[i][j] = img2[i-1][j]
                else:
                    if img2[i-1][j] == img2[i][j-1]:
                        img2[i][j] = img2[i][j-1]
                    else:
                        t_label = min(img2[i-1][j],img2[i][j-1])
                        img2[i][j] = t_label
                        if t_label == img2[i][j-1]:
                            equivalent_list.append([t_label,img2[i-1][j]])
                        if t_label == img2[i-1][j]:
                            equivalent_list.append([t_label,img2[i][j-1]])
                            
                    
    
    ds = DisjointSet()
    for pair in equivalent_list:
        ds.add(pair[0],pair[1])
        

    for i in range(0,length):
        for j in range(0,width):
            if img2[i][j] != 0:
                if img2[i][j] in ds.leader:
                    img2[i][j] = ds.leader[img2[i][j]]
               
                    
    for i in range(0,length):
        for j in range(0,width):
            if img2[i][j] != 0:
                if img2[i][j] not in label_coordinates:
                    label_coordinates[img2[i][j]] = {'Y':{i},'X':{j}}
                else:
                    label_coordinates[img2[i][j]]['Y'].add(i)
                    label_coordinates[img2[i][j]]['X'].add(j)
                    
    return label_coordinates


def refine(num):
    return num.strip('.').lstrip('0')

def create_and_store_as_csv(scores, answer_sheet):	
	with open(answer_sheet+'.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Q. No","a","b","c","d","e","Marks"])
	
		for i in range(0,9):
			writer.writerow([i+1] + scores[i*6:i*6+6])

		writer.writerow(["Total Marks"] + scores[60:66])



if __name__ =='__main__':  
	model = load_model('Model/MNISTmodel.h5')
	exam = sys.argv[1]
	answer_sheet =sys.argv[2]

	os.chdir("../Data/"+exam+'/'+answer_sheet)
	scores = []

	for i in range(7,84):
		if i%7 != 0:
			box_img = str(i)+'.png'
			print(box_img)
			img = cv2.imread(box_img, 1)
			img2 = img.copy()


			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#plt.title('Sampled Image')
			#plt.imshow(gray,cmap='Greys')
			#plt.show()

			#connected component labelling
			blur = cv2.GaussianBlur(gray , (5,5),0)

			binary_img = cv2.threshold(blur,0,1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
			binary_img = binary_img[5:-5,10:-10]
			kernel = np.ones((5, 5), np.uint8) 



			kernel_dilate = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype='uint8')
			binary_img = cv2.dilate(binary_img,kernel_dilate,iterations=1)

			#plt.title('Sample Image')
			#plt.imshow(binary_img,cmap='Greys')
			#plt.show()

			# plt.title('After Closing Image')
			# plt.imshow(binary_img,cmap='Greys')
			# plt.show()

			contours = connectedComponentLabelling(binary_img)


			#plt.title('Inverted Image')
			#plt.imshow(binary_img,cmap='Greys')
			#plt.show()
			Y_MIN = 10
			X_MIN = 2

			num = ''


			# pprint(contours)
			contours = sorted(contours.items(), key = lambda k_v: list(k_v[1]['X'])[0])

			# print(contours)    

			for label,contour in contours:
				x1 = min(contour['X'])
				x2 = max(contour['X'])
				y1 = min(contour['Y'])
				y2 = max(contour['Y'])
				if y2-y1 >= Y_MIN and x2-x1 >= X_MIN:
					roi = binary_img[y1:y2+1,x1:x2+1]
					num = num + str(number_recognition(roi,model))
				else:
					num = num + ''
			print('The number is',refine(num))
			scores.append(refine(num))

	#print(len(scores))
	#print(scores)
	create_and_store_as_csv(scores, answer_sheet)
	
	
	# print(numbers)
        
    
                    
