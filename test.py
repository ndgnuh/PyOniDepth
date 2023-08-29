import os

import cv2
import matplotlib
import numpy as np
import onidepth
from matplotlib.colors import LogNorm
from scipy.special import expit as sigmoid

# onidepth.hello()
matplotlib.use("tkagg")
from matplotlib import pyplot as plt

# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

# onidepth.hello_to("Hung")


# def correct_depth_map(dmap, min_depth, max_depth):
#     min_mask = dmap < min_depth
#     max_mask = dmap > max_depth
#     adj = 1 - (dmap.astype("float32") - min_depth) / (max_depth - min_depth)
#     adj = np.where(min_mask, np.ones_like(adj) * 0, adj)
#     adj = np.where(max_mask, np.ones_like(adj) * 1, adj)
#     # adj = np.where(max_mask, maxs, adj)
#     # adj = adj.astype("float32") / max_depth
#     return adj


def correct_depth_map(dmap, min_depth=250, max_depth=1000):
    mask = dmap == 0
    adj = max_depth - (dmap - min_depth)
    adj = np.where(mask, dmap, adj)
    return adj


def init(frame):
    im = plt.imshow(
        frame,
        cmap="gray",
    )
    plt.colorbar()
    return im


# animation function.  This is called sequentially
def animate(im, i):
    a = im.get_array()
    im.set_array(i)
    return im


cmap = plt.get_cmap("rainbow")
ok = onidepth.initialize()
cap = cv2.VideoCapture(0)
print("=" * 30)
print("init ok: ", ok)
im = None
while True:
    dframe = onidepth.get_frame()
    ok, cframe = cap.read()
    if not ok:
        continue

    # dframe2 = np.zeros(dframe.shape, dtype="uint8")
    # cv2.convertScaleAbs(dframe, dframe2, 255.0 / 8000)
    # dframe = dframe2
    # dframe = cv2.normalize(dframe.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)
    print(dframe.shape, dframe.min(), dframe.max())
    if im is None:
        im = init(dframe)
    else:
        animate(im, dframe)
    plt.ion()
    plt.clim(dframe.min(), dframe.max())
    plt.show(block=False)
    plt.pause(0.0001)

    cv2.imshow("depth", dframe)
    # cv2.imshow("frame", cframe)
    if cv2.waitKey(1) == 27:
        plt.figure()
        plt.imshow(dframe)
        plt.colorbar()
        np.save("dframe.npy", dframe)
        dframe = correct_depth_map(dframe, 250, 1000)

        plt.figure()
        np.save("dframe-corrected.npy", dframe)
        plt.imshow(dframe)
        plt.colorbar()
        plt.savefig("dframe-corrected.png")

        plt.figure()
        plt.imshow(cframe)
        plt.savefig("cframe.png")
        break


onidepth.destroy()
