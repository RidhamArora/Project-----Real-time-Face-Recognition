import cv2

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
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,0,0),2)
        

    cv2.imshow("Video Frame",frame)
    cv2.imshow("Video Gray Frame",gray_frame)
    
    #wait for the user to input -q then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()