import cv2 
import numpy as np
import math 
import sys
import os
import matplotlib.pyplot as plt

def get_lines(img,lines,thresh,rows) :
    retval = []
    diff = []
    count = 0
    for i in range(len(lines)-1) :
        diff.append(abs(lines[i][0] - lines[i+1][0]))
        
    print(diff)
    print(thresh)
    for i in range(len(diff)-1):
        if abs(diff[i] -diff[i+1]) < thresh:
            print(abs(diff[i] -diff[i+1]))
            retval.append(lines[i][0])
            retval.append(lines[i+1][0])
            count += 1

        elif count < rows:
            i = i
            count =0
            retval = []
            
    retval = list(dict.fromkeys(retval))
    retval.sort()
    
    
    final_answer = [ value for value in lines if value[0] in retval] 
    return final_answer

def intersection(r,theta,q,alpha):
    sint = np.sin(theta)
    sina = np.sin(alpha)
    cost = np.cos(theta)
    cosa = np.cos(alpha)
    y = (r*cosa - q*cost)//(sint*cosa - cost*sina)
    x = (r*sina - q*sint)//(cost*sina - cosa*sint)
    
    return [x,y]

def draw_lines(img, new_lines, thresh) :
    count =0
    prev = 0
    retval = []
    
    for line in new_lines:
        
        r = line[0]
        theta = line[1]
        
        
        if abs(abs(prev) - abs(r)) < thresh and count != 0:
            prev = r
            continue    
            
        prev = r
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*r
        y0 = b*r

        x1 = int(x0 - 1000*b)
        y1 = int(y0 + 1000*a)

        x2 = int(x0 + 1000*b)
        y2 = int(y0 - 1000*a)
        count +=1
        retval.append([r,theta])
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)

    return img, retval


def choose_lines(horizontal_lines, starting_thresh,no_of_rows):
    count = int(0)
    retval = []
    for line in horizontal_lines:
        if line[0] > starting_thresh:
            retval.append(line)
            count = count + 1
            
        if count == no_of_rows:
            break
        
    return retval


if __name__=='__main__':
	exam = sys.argv[1]
	filename = ''
	rows = 0
	no_of_rows = 13


	# Thresh values is an indication that the ones lesser than thresh will be removed so set acordingly
	if exam == 'Endsem':
		thresh = 20  #Indicates the width of the boxes
		thresh_draw = 20 #Indicates the width of the boxes
		rows = 9 #Indicates number of rows/number of lines to be considered
		starting_thresh = 150 #First horizontal line to be considered.
	else:
		thresh = 20
		thresh_draw = 12
		rows = 12
		starting_thresh = 115
		
	exam = "../Data/"+exam
	os.chdir(exam)
	for count,image_name in enumerate(os.listdir()):
		
		print(image_name)
		img = cv2.imread(image_name) 
		
		img = cv2.resize(img,(600,1200),interpolation = cv2.INTER_CUBIC)
		
		img_crop = img.copy()
		plt.title('The image ')
		plt.imshow(img)
		plt.show()
		# Convert the img to grayscale 
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

		# Apply edge detection method on the image 
		edges = cv2.Canny(gray,50,150,apertureSize = 3) 
		plt.title('The edges')
		plt.imshow(edges,cmap='gray')
		plt.show()

		# This returns an array of r and theta values 
		lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

		# The below for loop runs till r and theta values 
		# are in the range of the 2d array 
		horizontal_lines = []
		vertical_lines = []
		points = []
		for line in lines:
		    for r,theta in line:
		        if math.ceil(theta) == 2 :
		            horizontal_lines.append([r,theta])
		        else:
		            vertical_lines.append([r,theta])


		horizontal_lines.sort(key=lambda x: x[0])
		vertical_lines.sort(key=lambda x: abs(x[0]))
	

		img_copy = img.copy()
		img,horizontal_lines = draw_lines(img, horizontal_lines,thresh_draw)
		img, vertical_lines = draw_lines(img, vertical_lines, 16)
		
		horizontal_lines.sort(key=lambda x: x[0])
		vertical_lines.sort(key=lambda x: abs(x[0]))
		
		horizontal_lines = get_lines(img,horizontal_lines,thresh, rows)
		horizontal_lines = choose_lines(horizontal_lines,starting_thresh,no_of_rows)
		
		img_copy,horizontal_lines = draw_lines(img_copy, horizontal_lines , thresh_draw)
		img_copy, vertical_lines = draw_lines(img_copy, vertical_lines, 16)
		
		
		cv2.imwrite('linesdet'+str(count)+'.jpg', img_copy) 

		for i in range(len(horizontal_lines)):
		    points_temp = []
		    for j in range(len(vertical_lines)):
		        points_temp.append(intersection(horizontal_lines[i][0],horizontal_lines[i][1],vertical_lines[j][0],vertical_lines[j][1]))
		    points.append(points_temp)
		
		new_images = []
		new_dir = 'answer_sheet'+str(count)
		if new_dir not in os.listdir():
		    os.mkdir(new_dir)
		
		row = 0 
		for i in range(len(points)-1) :
		    for j in range(len(points[0])-1):
		        new_img = img_crop[int(points[i][j][1]) : int(points[i+1][j][1]), int(points[i][j][0]) : int(points[i][j+1][0])]
		        new_images.append([new_dir+str(row) + '.png', new_img])
		        cv2.imwrite(new_dir+'/'+str(row) + '.png', new_img)
		        row+=1

