import cv2 

# opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model\yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel()
model.setInputParams(size=(320,320),scale=1/255.)

#initialize camera 
cap=cv2.VideoCapture(0)

while True :
    # get the frames
    ret,frame = cap.read()

    # object detection
    (class_id,score,bboxes)=model.detect(frame)
    print("class ids :",class_id)
    print("scores :",score)
    print("bounding boxes:",bboxes)

    cv2.imshow("frame",frame)
    cv2.waitKey(1)