import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Datasets/Train"
TEST_DIR = "Datasets/Test"

LEARNING_RATE = 2e-5
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5 # 0.5*lambda_cycle
LAMBDA_MASK = 0.7
LAMBDA_CYCLE_MASK = 0.3
NUM_EPOCHS = 20
LOAD_MODEL = False
SAVE_MODEL = True
TRAIN_MODEL = True
TRANSFORMATION = "AppleToOrange"                          
TEST_MASK = "Random"

CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/HorseToZebra20/gen_a.pth.tar"     
CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/HorseToZebra20/gen_b.pth.tar"
CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/HorseToZebra20/disc_a.pth.tar"
CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints/AppleToOrange40/disc_b.pth.tar"
CHECKPOINT_DISC_AM = "../drive/MyDrive/Checkpoints/AppleToOrange40/disc_am.pth.tar"
CHECKPOINT_DISC_BM = "../drive/MyDrive/Checkpoints/AppleToOrange40/disc_bm.pth.tar"

NEW_CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/AppleToOrange30No2/gen_a.pth.tar"     
NEW_CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/AppleToOrange30No2/gen_b.pth.tar"
NEW_CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/AppleToOrange30No2/disc_a.pth.tar"
NEW_CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints/AppleToOrange30No2/disc_b.pth.tar"
NEW_CHECKPOINT_DISC_AM = "../drive/MyDrive/Checkpoints/AppleToOrange30No2/disc_am.pth.tar"
NEW_CHECKPOINT_DISC_BM = "../drive/MyDrive/Checkpoints/AppleToOrange30No2/disc_bm.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
