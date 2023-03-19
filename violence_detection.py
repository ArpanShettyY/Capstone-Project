from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import io
import ipywidgets
import random
import shutil


# Get the video names of violence and non-violence
non_violent=os.listdir("/content/Real Life Violence Dataset/NonViolence")
violent=os.listdir("/content/Real Life Violence Dataset/Violence")

# Splitting the entire dataset into 8:1:1 for train:validation:test
def split_test_train(lis):
  n=len(lis)
  other=random.sample(range(n),int(n/10))
  return ([x for i,x in enumerate(lis) if not i in other],[x for i,x in enumerate(lis) if i in other])

def split_train_valid(lis):
  n=len(lis)
  other=random.sample(range(n),int(n/9))
  return ([x for i,x in enumerate(lis) if not i in other],[x for i,x in enumerate(lis) if i in other])

train_non_violent,test_non_violent=split_test_train(non_violent)
train_violent,test_violent=split_test_train(violent)

train_non_violent,valid_non_violent=split_train_valid(train_non_violent)
train_violent,valid_violent=split_train_valid(train_violent)

# Moving the videos into the appropriate folder based on the split
src_path="/content/Real Life Violence Dataset/NonViolence/"
for filename in train_non_violent:
  shutil.move(os.path.join(src_path,filename), "/content/train")
for filename in test_non_violent:
  shutil.move(os.path.join(src_path,filename), "/content/test")
for filename in valid_non_violent:
  shutil.move(os.path.join(src_path,filename), "/content/valid")


src_path="/content/Real Life Violence Dataset/Violence/"
for filename in train_violent:
  shutil.move(os.path.join(src_path,filename), "/content/train")
for filename in test_violent:
  shutil.move(os.path.join(src_path,filename), "/content/test")
for filename in valid_violent:
  shutil.move(os.path.join(src_path,filename), "/content/valid")

train=[]
test=[]
valid=[]
for filename in train_non_violent:
  train.append([filename,"Non Violent"])
for filename in train_violent:
  train.append([filename,"Violent"])
for filename in test_non_violent:
  test.append([filename,"Non Violent"])
for filename in test_violent:
  test.append([filename,"Violent"])
for filename in valid_non_violent:
  valid.append([filename,"Non Violent"])
for filename in valid_violent:
  valid.append([filename,"Violent"])

# Store a dataframe storing the video name and its label(violent or non-violent)
train_df = pd.DataFrame(train, columns =["video_name","tag"])
test_df = pd.DataFrame(test, columns =["video_name","tag"])
valid_df=pd.DataFrame(valid, columns =["video_name","tag"])

# Randomly shuffle them
train_df=train_df.sample(frac = 1)
test_df=test_df.sample(frac = 1)
valid_df=valid_df.sample(frac = 1)

MAX_SEQ_LENGTH = 30 # The maximum number of frames extracted from each video
IMG_SIZE = 90 # the width/lenght of each frame extracted

# Layer for cropping the frame to the desired size
center_crop_layer = layers.CenterCrop(IMG_SIZE,IMG_SIZE)

def crop_center(frame):
    '''Crops the frame and outputs a numpy array'''
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

def load_video(path, max_frames):
    '''Extracts frames at fixed intervals throught the video till the video ends or the max_seq_lenght is reached'''
    frames_to_take_per_sec=5
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    k=int(fps/frames_to_take_per_sec)
    i=0
    j=0
    try:
      while j<max_frames:
        ret, frame = cap.read()
        if not ret:
          break
        if i%k==0:
          j+=1
          frame= crop_center(frame)
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frames.append(frame)
        i+=1
    finally:
      cap.release()
      return np.array(frames[:max_frames])





# Generates the label for the set of frames
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)



def prepare_all_videos(df, root_dir):
    '''Converts all the videos in the folder to the desired max_seq_lenght and img_size 
    along with their labels for further processing'''
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels).numpy()

    videos = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, IMG_SIZE,IMG_SIZE), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path),MAX_SEQ_LENGTH)

        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.empty((diff, IMG_SIZE, IMG_SIZE), dtype="float32")
            for d in range(diff):
              padding[d:]=frames[-1:]
            frames = np.concatenate((frames, padding))

        videos[idx,] = frames

    return videos, labels

train_videos, train_labels = prepare_all_videos(train_df,"/content/train")

test_videos, test_labels = prepare_all_videos(test_df,"/content/test")

valid_videos, valid_labels = prepare_all_videos(valid_df,"/content/valid")

BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1)
NUM_CLASSES = 1

# OPTIMIZER
LEARNING_RATE = 1e-4

# TRAINING
EPOCHS = 40

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8



@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


trainloader = prepare_dataloader(train_videos, train_labels, "train")
train_videos=[] 
train_labels=[]

validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
valid_videos=[] 
valid_labels=[]

testloader = prepare_dataloader(test_videos, test_labels, "test")
test_videos=[] 
test_labels=[]

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.relu),
                layers.Dense(units=embed_dim, activation=tf.nn.relu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="sigmoid")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)

    _, accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model


model = run_experiment()

def prepare_single_video(frames):
    '''Outputs the video in the desired img_size and seq_lenght'''
    if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.empty((diff, IMG_SIZE, IMG_SIZE), dtype="float32")
            for d in range(diff):
              padding[d:]=frames[-1:]
            frames = np.concatenate((frames, padding))


    return frames


def predict_action(path):
    '''Uses the model to predict whether the video has violence or not'''
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(path,MAX_SEQ_LENGTH)
    video = prepare_single_video(frames)
    pred = model.predict(tf.expand_dims(video, axis=0))[0]
    if pred>0.5:
      return class_vocab[1]
    else:
      return class_vocab[0]

# Checking the accuracy of the above model on occluded videos
v=0
for vid in os.listdir("/content/drive/MyDrive/Capstone Resources/data/vidsWithObstructions"):
  if predict_action("/content/drive/MyDrive/Capstone Resources/data/vidsWithObstructions/"+vid)=="Violent":
    v+=1
print("Accuracy on videos with occlusions",round(v*100/len(os.listdir("/content/drive/MyDrive/Capstone Resources/data/vidsWithObstructions")),2),"%")

# Checking the accuracy of the above model on collected indoor violence videos
v=0
for vid in os.listdir("/content/drive/MyDrive/Capstone Resources/Phase 1/Videos/Clips/"):
  if predict_action("/content/drive/MyDrive/Capstone Resources/Phase 1/Videos/Clips/"+vid)=="Violent":
    v+=1
print("Accuracy on videos with clips",round(v*100/len(os.listdir("/content/drive/MyDrive/Capstone Resources/Phase 1/Videos/Clips/")),2),"%")
