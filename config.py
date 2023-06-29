import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"              #Use cuda as device if available, else use the gpu
TRAIN_DIR = "Datasets/Train"                                        #Train Datasets Directory
TEST_DIR = "Datasets/Test"                                          #Test Datasets Directory

#Hyperparameters
LEARNING_RATE = 2e-5                                            
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5
LAMBDA_MASK = 0.7
LAMBDA_CYCLE_MASK = 0.3

#Settings
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
TRAIN_MODEL = True
TRANSFORMATION = "AppleToOrange"       
TRAIN_MASK = "Random"
TEST_MASK = "Random"

#Checkpoints to load
FOLDER = "AppleToOrange20No2"
CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/" + FOLDER + "/gen_a.pth.tar"     
CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/" + FOLDER + "/gen_b.pth.tar"
CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/" + FOLDER + "/disc_a.pth.tar"
CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints" + FOLDER + "/disc_b.pth.tar"
CHECKPOINT_DISC_AM = "../drive/MyDrive/Checkpoints/" + FOLDER + "/disc_am.pth.tar"
CHECKPOINT_DISC_BM = "../drive/MyDrive/Checkpoints/" + FOLDER + "/disc_bm.pth.tar"

#Checkpoints to save
NEW_FOLDER = "AppleToOrange30No2"
NEW_CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/gen_a.pth.tar"     
NEW_CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/gen_b.pth.tar"
NEW_CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/disc_a.pth.tar"
NEW_CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/disc_b.pth.tar"
NEW_CHECKPOINT_DISC_AM = "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/disc_am.pth.tar"
NEW_CHECKPOINT_DISC_BM = "../drive/MyDrive/Checkpoints/" + NEW_FOLDER + "/disc_bm.pth.tar"

#Transformations to apply to images
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
