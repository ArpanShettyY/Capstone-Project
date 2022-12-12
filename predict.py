'''
* Video is prepared by passing it to the frame 
  selection process.

* Selected frames are first pre-processed before 
  sending it to the model.

* The model receives these frames and then predicts the output.
'''

import tensorflow as tf
import cv2
import numpy as np

class Prediction:

    def __init__(self):
        try:
            self.model = tf.keras.models.load_model() # load tensorflow model here
        except:
            self.model = ""
        # hyperparameters for pre-processing
        self.MAX_SEQ_LENGTH = 30 
        self.IMG_SIZE = 90

    def resize_frames(self, frame):
        resize_layer = tf.keras.layers.Resizing(
            self.IMG_SIZE, self.IMG_SIZE, crop_to_aspect_ratio=True)
        resized = resize_layer(frame[None, ...])
        resized = resized.numpy().squeeze()
        return resized

    def load_video(self, frames):
        new_frames = []
        if not frames:
            return -1
        jump = 12
        start = 0
        while start < len(frames):
            frame = frames[start]
            frame = self.resize_frames(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_frames.append(frame)
            start += jump
        return new_frames[:self.MAX_SEQ_LENGTH]

    def prepare_single_video(self, frames):
        if len(frames) < self.MAX_SEQ_LENGTH:
            diff = self.MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, self.IMG_SIZE, self.IMG_SIZE))
            frames = np.concatenate((frames, padding))
        return frames

    def predict_action(self, frames):
        class_vocab = ['non-violent', 'violent']
        if self.model == "":
            return [-1, -1]
        frames = self.load_video(frames)
        if frames != -1:
            video = self.prepare_single_video(frames)
            pred = self.model.predict(tf.expand_dims(video, axis=0))[0]
            if pred > 0.5:
                #print(f"  {class_vocab[1]}")
                return [class_vocab[1], pred]
            else:
                #print(f"  {class_vocab[0]}")
                return [class_vocab[0], pred]
        else:
            return [-1, -1]