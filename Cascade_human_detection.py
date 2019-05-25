import cv2
import time
import os
import glob
t = time.time() * 100
sessionid = "rowan-" + str(t)
if not os.path.exists(sessionid):
    os.makedirs(sessionid)

# person_cascade = cv2.CascadeClassifier(
#     os.path.join('./HS.xml'))
person_cascade = cv2.CascadeClassifier(
    os.path.join('./haarcascade_upperbody.xml'))
# person_cascade = cv2.CascadeClassifier(
#     os.path.join('./haarcascade_fullbody.xml'))
#cap = cv2.VideoCapture("./apollo_camera/apollo_test_3.mp4")

password = "paas2019!@1"
# cap = cv2.VideoCapture('rtsp://testaccount:Paaspop2019@192.168.190.188')
# cap = cv2.VideoCapture('test.mp4')
# cap = cv2.VideoCapture(0)

i = 0

# for filename in glob.glob('./VisorImages/*.jpg'):
for filename in sorted(os.listdir('./rowan')):
    if filename != ".DS_Store":
        frame = cv2.imread("./rowan/"+filename)
        (h, w) = frame.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)
        angle180 = 180
        scale = 1.0
        # 180 degrees
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        frame = cv2.warpAffine(frame, M, (w, h))

        # while True:
    #     r, frame = cap.read()
        if True:#r:
            # start_time = time.time()
            frame = cv2.resize(frame,(640,360)) # Downscale to improve frame rate
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Haar-cascade classifier needs a grayscale image
            rects = person_cascade.detectMultiScale(gray_frame)

            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
            # out.write(frame) # write the frame to the output
            cv2.imwrite('./'+ sessionid +'/opencv' + str(i) + '.png', frame)
            i += 1
            cv2.imshow("preview", frame)
            # end_time = time.time()
            # print("Elapsed Time:",end_time-start_time)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"): # Exit condition
            break

# cap.release()

img_array = []
# size = (640,360)
# for filename in glob.glob('./'+sessionid+'/*.png'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
#
# out = cv2.VideoWriter(sessionid+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
path = "./" + str(sessionid) + "/"
for count in range(len(os.listdir(path))):
    filename = './'+sessionid+'/opencv' + str(count) + '.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

#seems like AVI is the ONLY cross platform supported codec. Not that others cannot work but its the only supported.
out = cv2.VideoWriter(sessionid+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
# out = cv2.VideoWriter(sessionid+'.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, size)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

cv2.destroyAllWindows()