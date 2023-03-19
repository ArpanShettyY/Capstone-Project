# Capstone-Project
Indoor Violence Detection and Alert system


# Create directories to store the training, testing and validation data respectively
!mkdir train
!mkdir test
!mkdir valid

# Downloading the RLVS dataset
!kaggle datasets download mohamedmustafa/real-life-violence-situations-dataset -p /content/ --unzip

! pip install -q kaggle

!mkdir ~/kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json
