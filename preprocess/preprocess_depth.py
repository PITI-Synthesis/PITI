from PIL import Image
import blobfile as bf
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

data_dir = "/mnt/blob/dataset/DIODE/val.txt"


def plot_depth_map(dm, validity_mask, name):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    norm = plt.Normalize(vmin=0, vmax=np.log(MAX_DEPTH + 1.01))
    image = cmap(norm(dm))
    plt.imsave(name, np.clip(image, 0., 1.))

def plot_normal_map(normal_map, name):
    normal_viz = normal_map[:, ::, :]

    normal_viz = normal_viz + np.equal(np.sum(normal_viz, 2, 
    keepdims=True), 0.).astype(np.float32)*np.min(normal_viz)

    normal_viz = (normal_viz - np.min(normal_viz))/2.
    plt.axis('off')
    plt.imsave(name, np.clip(normal_viz, 0., 1.))


with open(data_dir) as f:
        all_files = f.read().splitlines() 

for file in all_files:
        print(file)
        path = os.path.join("/mnt/blob/dataset/DIODE/" , file)
        pil_image = Image.open(path).convert("RGB").resize((256,256), Image.BICUBIC)
        

        path2 = os.path.join("/mnt/blob/dataset/DIODE/", file[:-4]+"_depth.npy")
        depth = np.load(path2).squeeze()
        depth = depth.astype(np.float32)

        path3 = os.path.join("/mnt/blob/dataset/DIODE/", file[:-4]+"_depth_mask.npy") 
        depth_mask = np.load(path3)
        depth_mask = depth_mask.astype(np.float32)

 
        path4 = os.path.join("/mnt/blob/dataset/DIODE/", file[:-4]+"_normal.npy") 
        normal = np.load(path4)
        normal = normal.astype(np.float32)
 
        image_depth = cv2.resize(depth, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        image_depth_mask = cv2.resize(depth_mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
 
        normal = cv2.resize(normal, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
  
        name = os.path.join("/mnt/blob/dataset/DIODE-256/val" , file.split('/')[-1])
        pil_image.save(name)
        plot_depth_map(image_depth, image_depth_mask, name[:-4]+"_depth.png")
        plot_normal_map(normal, name[:-4]+"_normal.png")
       
