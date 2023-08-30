import os

import cv2
import matplotlib
import numpy as np
import onidepth
from matplotlib.colors import LogNorm
from retinaface_detector import FaceDetector
from scipy.special import expit as sigmoid

detector = FaceDetector()
# onidepth.hello()
matplotlib.use("tkagg")
from matplotlib import pyplot as plt


def init(frame):
    im = plt.imshow(frame)
    plt.colorbar()
    return im


# animation function.  This is called sequentially
def animate(im, i):
    a = im.get_array()
    im.set_array(i)
    return im


cmap = plt.get_cmap("rainbow")
max_depth = 0
min_depth = float("inf")
ok = onidepth.initialize(
    width=640,
    height=400,
    fps=30,
    pixel_format=onidepth.PIXEL_FORMAT_DEPTH_1_MM,
)
onidepth.set_min_depth(250)
onidepth.set_max_depth(1000)

if not ok:
    import sys

    sys.exit(0)
cap = cv2.VideoCapture(0)
print("=" * 30)
print("init ok: ", ok)
im = None


# Normalize the grayscale heatmap to [0, 1]
def gen_colormap(hmap, norm=False):
    colormap = plt.cm.viridis
    if norm:
        nhmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    else:
        nhmap = hmap
    return colormap(nhmap)


def gen_histmap(hm):
    hist = np.zeros(hm.shape)
    total = np.count_nonzero(hm)
    prev_count = 0
    for v in np.unique(hm):
        if v > 0:
            mask = hm == v
            count = np.count_nonzero(mask)
            hist[mask] = count = (total - count - prev_count) / total
            prev_count += count

    return hist


def gen_mask(dmap):
    a = dmap.min()
    b = dmap.max()
    r = 10
    out = np.zeros_like(dmap, dtype="float32")
    c = 1 / r
    for i in np.linspace(a, b + c, r):
        print(i)
        out[out >= i] = c * i
    return out


while True:
    dframe = onidepth.get_frame()
    ok, cframe = cap.read()
    if not ok:
        continue

    # Detect faces
    boxes, scores, landmarks = detector(cframe)
    print(boxes)
    H, W, _ = cframe.shape
    if len(boxes) > 0 and False:
        x1, y1, x2, y2 = boxes[0]
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        cv2.rectangle(cframe, (x1, y1), (x2, y2), (0, 255, 0))

        H, W = dframe.shape
        x1, y1, x2, y2 = boxes[0]
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        mask = np.zeros_like(dframe, dtype=bool)
        mask[y1:y2, x1:x2] = True

        dframe = dframe.astype("float32")
        try:
            dmax = dframe[mask].max()
            dmin = dframe[mask].min()
            dframe = (dframe - dmin) / (dmax - dmin)
        except ValueError:
            pass
        dframe = np.where(mask, dframe, np.zeros_like(dframe))

    dframe = gen_colormap(dframe)

    # gen_hist(dframe)
    # break
    # dframe2 = np.zeros(dframe.shape, dtype="uint8")
    # cv2.convertScaleAbs(dframe, dframe2, 255.0 / 8000)
    # dframe = dframe2
    # dframe = cv2.normalize(dframe.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)
    # print(dframe.shape, dframe.min(), dframe.max())
    if im is None:
        im = init(dframe)
    else:
        animate(im, dframe)
    plt.ion()
    # print(min_depth, max_depth)
    max_depth = max(max_depth, dframe.max())
    min_depth = min(min_depth, dframe.min())
    # plt.colorbar()
    plt.clim(min_depth, max_depth)
    plt.show(block=False)
    plt.pause(0.0001)

    cv2.imshow("", cframe)
    # cv2.imshow("frame", cframe)
    if cv2.waitKey(1) == 27:
        # plt.figure()
        # plt.imshow(dframe)
        # plt.colorbar()
        # np.save("dframe.npy", dframe)
        # dframe = correct_depth_map(dframe, 250, 1000)

        # plt.figure()
        # np.save("dframe-corrected.npy", dframe)
        # plt.imshow(dframe)
        # plt.colorbar()
        # plt.savefig("dframe-corrected.png")

        # plt.figure()
        # plt.imshow(cframe)
        # plt.savefig("cframe.png")
        break


onidepth.destroy()
