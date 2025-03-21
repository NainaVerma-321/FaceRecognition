import cv2

#face_cap=cv2.CascadeClassifier("C:/Usersshwet/AppDataLocal/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#camera open code 


video_cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video_cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data,cv2.COLOR_BGRA2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags =cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data)
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    cv2.imshow("video_live", video_data)

    # Wait for 'a' key press to break the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the camera after the loop ends
video_cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


