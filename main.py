import cv2 

# opencv DNN

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model\yolov4-tiny.cfg")

cap=cv2.VideoCapture(0)

while True :
    ret,frame = cap.read()

    cv2.imshow("frame",frame)
    cv2.waitKey(1)