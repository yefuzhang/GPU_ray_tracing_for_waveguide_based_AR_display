import gdown
import numpy as np
import os

def download_and_load(file_id, local_name):
    """Download from Google Drive (if needed) and load npy"""
    if not os.path.exists(local_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {local_name} ...")
        gdown.download(url, local_name, quiet=False)
    return np.load(local_name, allow_pickle=False)

# Replace each with your actual file_id from Google Drive
lut_ic1 = download_and_load("1HiBhh3sw_5FW5Ylm0jLJ_3zmGPG0ibIN", "lut_ic1_fullColor.npy")
lut_ic2 = download_and_load("13gRhhL6G-nojuwibtlxt3BOZijd5kXVP", "lut_ic2_fullColor.npy")
lut_ic3 = download_and_load("1Zyy8lzUKki2iQ-u2F9JhMxsXbPI-M-YD", "lut_ic3_fullColor.npy")
lut_fc1 = download_and_load("1wLNyFuBMWr2q3UtPI5FZy8GE5TPGaHGg", "lut_fc1_fullColor.npy")
lut_fc2 = download_and_load("1MDjkBDgcs_YssEb6RHQtQJ6iWEqo_aGd", "lut_fc2_fullColor.npy")
lut_oc1 = download_and_load("1WJZvcRpYeMwYBPxSLQewQXERYUAKD0in", "lut_oc1_fullColor.npy")
lut_oc2 = download_and_load("11SCZNpk0bcX7tM_ihbsz-YSsRudMIrRB", "lut_oc2_fullColor.npy")

print("All LUTs loaded successfully!")
