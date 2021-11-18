#! /usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

import vuba
import heartcv as hcv

import sparse_oflow
import util

small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

# Parameters ------------------------------------
# params for Harris corner detection
feature_params = dict(
    maxCorners=500,
    qualityLevel=0.001,
    minDistance=0,
    blockSize=7,
    useHarrisDetector=0,
    k=0.4,
)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

file = "./test.avi"
split_by = 600 # acquisition length

# Create the mask
video = vuba.Video(file)
frames = video.read(0, len(video), low_memory=False, grayscale=True)
frames = [vuba.shrink(frame, by=50) for frame in frames]

# Run analysis -----------------------------------
writer = vuba.Writer('./test_out.avi', video, resolution=(video.width*3, video.height), codec='MJPG')
pm, nm, ce, cm, pa, px, py, na, nx, ny = sparse_oflow.motion_tracks(frames, None, feature_params, lk_params, writer, display=False)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

ax1.plot(cm, label='Center of mass (distances)')
ax1.legend(loc='lower right')

ax2.plot(pm, label='Positive rotation (distances)')
ax2.plot(nm, label='Negative rotation (distances)')
ax2.legend(loc='lower right')

ax3.plot(pa, label='Positive rotation (angles)')
ax3.plot(na, label='Negative rotation (angles)')
ax3.legend(loc='lower right')

ax4.plot(px, label='Positive rotation (x)')
ax4.plot(nx, label='Negative rotation (x)')
ax4.plot(py, label='Positive rotation (y)')
ax4.plot(ny, label='Negative rotation (y)')
ax4.legend(loc='lower right')

plt.show()

video.close()
