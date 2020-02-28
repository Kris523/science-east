import os
import cv2
from base_camera import BaseCamera
import numpy as np

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)

        ##DNN
        prototxt = 'deploy.prototxt.txt'
        model = 'res10_300x300_ssd_iter_140000.caffemodel'
        detection_confidence = 0.75
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        faceRecentlyFound=False
        faceIndex = 0
        ##

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            ##DNN
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            #draw Boxes:
            faceinFrame = False
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > detection_confidence:
                    faceinFrame = True
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(img, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(img, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    if(not faceRecentlyFound):
                        print("rising edge" + str(faceIndex))
                        cv2.imwrite("static/{}.png".format(faceIndex), img[startY:endY, startX:endX])
                        faceIndex = faceIndex + 1
                        faceRecentlyFound = True

            if(faceRecentlyFound and not faceinFrame):
                faceRecentlyFound = False
                print("falling edge")
            ##


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()



        #### This is the OpenCV version. Instead, lets try running a model.
        #cascPath = "haarcascade_frontalface_default.xml"
        #faceCascade = cv2.CascadeClassifier(cascPath)
        ####





            #### This is the OpenCV version. Instead, lets try running a model.
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = faceCascade.detectMultiScale(
            #     gray,
            #     scaleFactor=1.5,
            #     minNeighbors=5,
            #     minSize=(30, 30),
            #     flags=cv2.CASCADE_SCALE_IMAGE
            # )
            #
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #####