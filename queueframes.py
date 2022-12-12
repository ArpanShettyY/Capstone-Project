'''
* class used to get the frames from the video feed
  in the form of frames.

* this class imitates a sliding window.

* the list of selected frames is 300 in length.

* self.right value and self.left value increases 
  everytime a frame has been added.
  
* when length of list containing the frames reaches 300,
  the first frame is discarded and a new frame is added to
  the end of the list.
'''

class queueFrames:

    def __init__(self):
        self.frames = []
        self.right = 0
        self.left = 0
        self.framesLength = 0

    def getLength(self):
        return self.framesLength

    def getFrames(self):
        return self.frames

    def addToQueue(self, frame):
        if len(self.frames) == 300:
            self.left += 1
            self.framesLength = self.framesLength - 1
            self.frames = self.frames[1:self.framesLength+2]
        self.frames.append(frame)
        self.right += 1
        self.framesLength += 1