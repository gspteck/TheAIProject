import cv2

#load pre-trained data on face frontals (downloaded from the opencv repository on github)
trainedFaceData = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

#choose an image to detect the face from
img = cv2.imread("./faces/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")

#make the image black and white
grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect the faces
faceCoordinates = trainedFaceData.detectMultiScale(grayscaleImg)

#draw rectangle around the faces
for i in faceCoordinates:
    cv2.rectangle(img, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 255, 0), 2)

cv2.imshow("Face Detector", img)
cv2.waitKey()