import cv2
import numpy as np
import os

def dist(X,Y):
    return np.sqrt(sum((X-Y)**2))

def KNN(X,Y,query_point,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(query_point,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    
    vals = vals[:k]
    vals = np.array(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return int(pred)
    
skip=0
face_data = []
labels = []
dataset_path = './data/'
class_id = 0
names = {}

#Preparing the data
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print(fx)
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        data_item = data_item[:11,:]
        face_data.append(data_item)
        
        #Create Labels for the Class
        target = class_id*np.ones((data_item.shape[0]))
        class_id+=1
        labels.append(target)
        
face_dataset = np.concatenate(face_data,axis=0)
face_labels  = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue
    
    #5 is the no. of neighbrours.
    #Higher the value of it , lesser are the no. of detections
    #But with Higher Quality
    #3-6 is optimal number
    #1.3 is the scaling factor which leads to shrinkage og image so that it is applied
    #to the same size as that of training data
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    for (x,y,w,h) in faces:
        
        #Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+h+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        out = KNN(face_dataset,face_labels,face_section.flatten(),20)
        
        pred_name = names[out]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.imshow("Video Frame",frame)
    
        #wait for the user to input -q then you will stop the loop
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break
    
    
cap.release()
cv2.destroyAllWindows()
        
        