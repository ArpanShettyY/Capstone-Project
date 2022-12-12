import cv2
from predict import Prediction
from queueframes import queueFrames
from notify import Notify

keyPressed = 0

cap = cv2.VideoCapture(0)

# frames to be saved as video in this format
output = cv2.VideoWriter("cam_video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 20, (640,480))

queue = queueFrames()

print("*** Initiating Prediction model ***")
prediction = Prediction()

print("*** Initiating notification manager ***")
notify = Notify()

frameCount = 0

while True:
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, (640, 480))
        queue.addToQueue(img)
        cv2.imshow("capture", img)
        frameCount+=1
        if queue.getLength() == 300 and frameCount%10==0:
            pred = prediction.predict_action(queue.getFrames())
            action, acc = pred[0], pred[1]
            print(action, acc)
            if action == -1:
                print("Frames/Model not loaded properly")
                cap.release()
                break
            if action == "violent":
                print("Violence detected. Pred acc :", acc[0])
                cv2.destroyAllWindows()
                frames = queue.getFrames()
                prediction.create(frames, output)
                notify.send()
                cap.release()
                break
    
    else:
        cv2.destroyAllWindows()
        cap.release()
        break

    if cv2.waitKey(1) & 0XFF == ord('x'):
        keyPressed = 1
        cv2.destroyAllWindows()
        cap.release()
        break

# predict action in case video has less than 300 frames
if not keyPressed:
    pred = prediction.predict_action(queue.getFrames())
    action, acc = pred[0], pred[1]
    print(action, acc)