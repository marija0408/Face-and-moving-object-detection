import cv2, time

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("face_detection/haarcascade_frontalface_default.xml")


while True:
    check, frame = video.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image,scaleFactor = 1.1,minNeighbors = 5 )


    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(175,35,78),2)



    cv2.imshow("Capturing",frame)

    key = cv2.waitKey(1)
    if key == ord('q'): #press q to end the session
        break


video.release()
cv2.destroyAllWindows