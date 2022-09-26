from cmath import nan
import cv2, time 
import pandas as pd
from datetime import datetime

first_frame = None

video = cv2.VideoCapture(0)

df = pd.DataFrame(columns = ["object_id", "start_time", "end_time"])

object_id = 0
prev_status = 0

while True:

    status = 0

    check, frame = video.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #removes noise and increases accuracy of a calculation
    gray = cv2.GaussianBlur(gray,(21,21),0) 

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)

    #black numpy has value 0, white has 255

    thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    #30 is the difference of black-white color which represents the tresshold for object detection
    #if it's greater than 30, something is moving

    #all values above 30 are colored to color 255 (white)
    #thresh binary we are using, but there are other methods which we can try

    #remove black holes in white image
    thresh_frame = cv2.dilate(thresh_frame,None, iterations=2)

    #object contour search 
    (cnts,_) =cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #we are sending a copy of tresh_frame
    #RETR_EXTERNAL 
    #CHAIN_APPROX - approximation method for contour retrieval 

    
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(136,10,10),2)

    if prev_status == 0 and status == 1:
        df.loc[object_id] = [object_id, datetime.now(), nan]
        
    if prev_status == 1 and status == 0:
        df.loc[df.object_id == object_id, "end_time"] = datetime.now()
        object_id = object_id + 1


    cv2.imshow("Original", frame)
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Thresh frame",thresh_frame)
    
    prev_status = status

    key = cv2.waitKey(1)
    if key == ord('q'): 
        if status == 1:
            df.loc[df.object_id == object_id, "end_time"] = datetime.now()
        break

print(cnts)
print(df)
df.csv('Object_detection_times.csv')
video.release()
cv2.destroyAllWindows