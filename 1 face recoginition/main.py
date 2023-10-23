print("hello")
import cv2
import face_recognition as fc
import numpy as np
import os

images = []
classNames = []
mylist = os.listdir(('images'))

for cls in mylist:
    curImg = cv2.imread(f'images/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


print(classNames)
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode=fc.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
#        
    return encodeList

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgs = cv2.resize(img , (0,0) , None , 0.25 , 0.25)
    imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)
    facesCurFrame = fc.face_locations(imgs)
    encodesCurFrame = fc.face_encodings(imgs , facesCurFrame)

    for encodeFace , faceLoc in zip(encodesCurFrame , facesCurFrame):
        matches = fc.compare_faces(encodeListKnown , encodeFace)
        faceDis = fc.face_distance(encodeListKnown , encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1 , x2 ,y2 ,x1 = faceLoc
            x1 , x2 ,y1 ,y2 = x1*4 , x2*4 , y1*4 , y2*4
            cv2.rectangle(img , (x1,y1), (x2,y2) , (0,255,0) , 2)
            cv2.rectangle(img , (x1 , y2-35) , (x2,y2) , (0 ,255 ,0) , cv2.FILLED)
            cv2.putText(img ,name , (x1+6 , y2-6) , cv2.FONT_HERSHEY_COMPLEX ,1,(255 ,255 ,255) , 2)



    cv2.imshow('webcam' , img)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break




# imgM = fc.load_image_file('images\mayank.jpg')
# imgM = cv2.cvtColor(imgM , cv2.COLOR_BGR2RGB)

# imgTest = fc.load_image_file('images\mayank_test.jpg')
# imgTest = cv2.cvtColor(imgM , cv2.COLOR_BGR2RGB)





