'''
This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import os
import glob
import argparse
import numpy as np
from PIL import Image

from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama

import time
import torch
import cv2
from typing import Union
from torchvision import transforms
from equilib.equi2equi import TorchEqui2Equi

def preprocess(
    img: Union[np.ndarray, Image.Image],
    is_cv2: bool = False,
) -> torch.Tensor:
    r"""Preprocesses image"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image):
        # Sometimes images are RGBA
        img = img.convert("RGB")

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img = to_tensor(img)
    assert len(img.shape) == 3, "input must be dim=3"
    assert img.shape[0] == 3, "input must be HWC"
    return img


def postprocess(
    img: torch.Tensor,
    to_cv2: bool = False,
) -> Union[np.ndarray, Image.Image]:
    if to_cv2:
        img = np.asarray(img.to("cpu").numpy() * 255, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        to_PIL = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
        )
        img = img.to("cpu")
        img = to_PIL(img)
        return img


t_start = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--rgbonly', action='store_true',
                    help='Add this if use are preparing customer dataset')
# Preprocessing related arguments
parser.add_argument('--q_error', default=0.7, type=float)
parser.add_argument('--refine_iter', default=3, type=int)
args = parser.parse_args()

paths = sorted(glob.glob(args.img_glob))
if len(paths) == 0:
    print('no images found')

# Check given path exist
for path in paths:
    assert os.path.isfile(path), '%s not found' % path

# Check target directory
if not os.path.isdir(args.output_dir):
    print('Output directory %s not existed. Create one.')
    os.makedirs(args.output_dir)

# Process each input
for i_path in paths:
    print('Processing', i_path, flush=True)

    # Load and cat input images
    #img_ori = np.array(Image.open(i_path).resize((1024, 512), Image.BICUBIC))[..., :3]
    img_src = Image.open(i_path)
    width, height = img_src.size
    img_small = np.array(img_src.resize((1024,512), Image.BICUBIC))[..., :3]
    #img_ori = np.array(img_src)[..., :3]

    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_small,
                                                    qError=args.q_error,
                                                    refineIter=args.refine_iter)

    # Align images with VP
    #i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])

    vp =  vp[2::-1]
    R = np.linalg.inv(vp.T)
    euler, _ = cv2.Rodrigues(R)
    rot = {
        "roll": euler[0,0] ,  #
        "pitch":euler[1,0],  # vertical
        "yaw": euler[2,0],  # horizontal
    }
    device = torch.device("cpu")
    img_src = preprocess(img_src).to(device)

    # Initialize equi2equi
    equi2equi = TorchEqui2Equi(h_out=height, w_out=width)
    out_img = equi2equi(
        src=img_src,
        rot=rot,
        sampling_method="torch",
        mode="bilinear",
    )
    out_img = postprocess(out_img)

    #panoEdge = (panoEdge > 0)
    #l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    if True: #args.rgbonly:
        path = os.path.join(args.output_dir, '%s.jpg' % basename)
        #Image.fromarray((i_img * 255).astype(np.uint8)).save(path)
        out_img.save(path)
    else:
        path_VP = os.path.join(args.output_dir, '%s_VP.txt' % basename)
        path_i_img = os.path.join(args.output_dir, '%s_aligned_rgb.png' % basename)
        path_l_img = os.path.join(args.output_dir, '%s_aligned_line.png' % basename)

        with open(path_VP, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
        #Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)

print('time spent: {}'.format(time.time() - t_start))
