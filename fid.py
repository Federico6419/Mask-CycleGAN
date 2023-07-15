import sys
sys.path.append('/content/Mask-CycleGAN/FID/src')
from pytorch_fid.fid_score import main

def compute_fid(path1, path2):
    fid_value = main(path1, path2)
    return fid_value
    
