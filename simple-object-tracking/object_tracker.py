# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# TO-DO
# see how the detection box is outputted from the detection model (topleft, topright etc.)
# locate where the frame is outputted and the next frame is looked at
# add information to a dictionary about center!! of the object detected for that frame, and use the assigned ID from the centroidtracker to store the information

# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


prototxt = "C:/Users/20184364/Downloads/simple-object-tracking/deploy.prototxt"
model = "C:/Users/20184364/Downloads/simple-object-tracking/res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5
vs = cv2.VideoCapture("C:/Users/20184364/Downloads/simple-object-tracking/video.avi")

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(src="C:/Users/20184364/Downloads/simple-object-tracking/video.avi").start()
#time.sleep(2.0)

# loop over the frames from the video stream
#while (vs.isOpened() == False):
while True:
    # while True:
    # read the next frame from the video stream and resize it
    ret, frame = vs.read()
    if frame is None:
        from centroidtracker import info_df
        print(info_df + "FIXED")
    else:
        frame = imutils.resize(frame, width=400)
     # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            print(detections.shape)

            #file = "frame_information"

            #if not os.path.exists(file):
            #    os.makedirs(file)



    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "Person {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # add information to the list
    #{"frame number", "detected_object", "topleft", "topright", "botleft", "botright"}
    #row = {ID, center_for_ID}
    #frame_info.append(row)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
print(info_df)
# vs.stop()
