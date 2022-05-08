import json
import math
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path as osp
import random
import sys
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from perception.webcam_sensor import WebcamSensor
from autolab_core import ColorImage, Image, PointCloud, RigidTransform, camera_intrinsics
from phoxipy import PhoXiSensor
from perception.colorized_phoxi_sensor import ColorizedPhoXiSensor

from sym.config import CI, CM


class ShapeNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/{split}.txt", dtype=str)
        random.seed(0)
        random.shuffle(filelist)
        filelist = [f for f in filelist if "03636649" not in f]  # remove lamps
        if (
            split == "train"
            and hasattr(CI, "only_car_plane_chair")
            and CI.only_car_plane_chair
        ):
            filelist = [
                f
                for f in filelist
                if "02691156" in f or "02958343" in f or "03001627" in f
            ]
        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist2 = filelist
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        image = cv2.imread(f"{prefix}.png", -1).astype(np.float32) 
        # import pdb;pdb.set_trace() 
        image = image / 255.0
        # plt.imshow(image)
        # plt.show()
        image = np.rollaxis(image, 2).copy()

        depth = cv2.imread(f"{prefix}_depth0001.exr", -1).astype(np.float32)[:, :, 0:1]

        depth[depth > 20] = 0
        plt.imshow(depth)
        plt.colorbar()
        plt.show()
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy() # (256, 256, 1) => (1, 256, 256)
        
        
        with open(f"{prefix}.json") as f:
            js = json.load(f)
        Rt, K_ = np.array(js["RT"]), np.array(js["K"])
        K = np.eye(4)
        K[:3, :3] = K_[np.ix_([0, 1, 3], [0, 1, 2])]
        # K[:3] *= -1
        # print(Rt, K)
        KRt = K @ Rt

        oprefix = self.filelist2[idx]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        depth_scale = 1 / abs(Rt[2][3])

        ########### Take depth as input ##################
        depth_noisy = depth.copy()
        depth_noisy += np.random.normal(0, 0.1*abs(Rt[2][3]), depth.shape)
        image_with_depth = np.concatenate((image, depth_noisy), axis=0)
        # plt.imshow(np.rollaxis(image_with_depth, 0, 3)[:, :, :4])
        # plt.show()
        ########### Take depth as input ##################

        S0 = [
            KRt @ np.diagflat([1, -1, 1, 1]) @ LA.inv(KRt),
            KRt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(KRt),
            KRt @ np.diagflat([-1, -1, 1, 1]) @ LA.inv(KRt),
        ]
        result = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            # "image": torch.tensor(image_with_depth).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "K": torch.tensor(K).float(),
            "RT": torch.tensor(Rt).float(),
        }

        w0, ws = sample_plane(Rt)
        S = [K @ w2S(w) @ LA.inv(K) for w in ws]
        y = [to_label(w, w0) for w in ws]
        result["S"] = torch.tensor(S).float()
        result["y"] = torch.tensor(y).float()
        result["w"] = torch.tensor(ws).float()
        result["w0"] = torch.tensor(w0).float()

        return result


class Pix3dDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split
        with open(f"{rootdir}/pix3d_info.json", "r") as fin:
            data_lists = json.load(fin)
        data_valid = set(np.loadtxt(f"{rootdir}/pix3d-valid.txt", dtype=str))

        data_lists = [
            d
            for d in data_lists
            if not d["truncated"] and not d["occluded"] and d["img"][:-4] in data_valid
        ]
        random.seed(0)
        random.shuffle(data_lists)
        if self.split == "train":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 != 0]
            self.size = len(data_lists) * 2
        elif self.split == "valid":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 == 0]
            self.size = len(data_lists)
        else:
            raise NotImplementedError

        self.data_lists = data_lists

        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        do_flip = False
        if self.split == "train":
            do_flip = idx % 2 == 1
            idx //= 2

        data_item = self.data_lists[idx]
        fimage = osp.join(self.rootdir, data_item["img"])
        fdepth = osp.join(self.rootdir, data_item["depth"])
        fmask = osp.join(self.rootdir, data_item["mask"])

        image = cv2.imread(fimage, -1).astype(np.float32)/255.0
        image = np.rollaxis(image, 2).copy()
        mask = cv2.imread(fmask).astype(np.float32)[None, :, :, 0] / 255.0
        image = np.concatenate([image, mask])
        depth = cv2.imread(fdepth, -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0

        depth = np.rollaxis(depth, 2).copy()
        Rt, K = np.array(data_item["Rt"]), np.array(data_item["K"])
        K[:3] *= -1

        if do_flip:
            K = np.diagflat([-1, 1, 1, 1]) @ K
            image = image[:, :, ::-1].copy()
            depth = depth[:, :, ::-1].copy()

        KRt = K @ Rt
        S0 = [KRt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(KRt)]

        w0, ws = sample_plane(Rt, np.array([1, 0, 0, 0]))
        S = [K @ w2S(w) @ LA.inv(K) for w in ws]
        y = [to_label(w, w0) for w in ws]
        depth_scale = 1 / (w0 @ Rt[:3, 3])

        # print(S0)
        # print("Rt", Rt)
        # print("S0", Rt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(Rt))
        # print("S0", w2S(w0))
        # print("w0", w0)
        # print("depth_scale", depth_scale)
        S0 = [K @ w2S(w0) @ LA.inv(K)]
        # print(S0)

        oprefix = data_item["mask"][5:-4]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        result = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "S0": torch.tensor(S0).float(),
            "K": torch.tensor(K).float(),
            "RT": torch.tensor(Rt).float(),
        }
        result["S"] = torch.tensor(S).float()
        result["y"] = torch.tensor(y).float()
        result["w"] = torch.tensor(ws).float()
        result["w0"] = torch.tensor(w0).float()

        return result

class RealDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        self.camera = PhoXiSensor("1703005")
        self.camera.start()
        self.camera._set_intr()
        filelist = glob(f"{rootdir}/*.npy")
        #Get all the items that have "segmented" in its file name.
        filelist = [s for s in filelist if "segmented" in s]
        print(filelist)
        self.filelist = filelist
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]

        # image = cv2.imread(f"{prefix}", -1).astype(np.float32)
        image = np.load(f"{prefix}", allow_pickle=True)
        print("png taken in image shape", image.shape)
        depth = image[:,:,3:]
        image = image[:,:,:3]
        # import pdb;pdb.set_trace()
        # image[:,:,-1] = image[:,:,-1] * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) 
        image[:,:,:3] = image[:,:,:3] / 255.0
        print(image.shape)
        plt.imshow(image)
        plt.show()

        ############# if also feed in depth as input ############
        # image = np.concatenate((image, depth), axis=2)
        # print(image.mean(axis=2))
        #########################################################

        image = np.rollaxis(image, 2).copy()
        print(image.shape)

        # K = np.identity(4)
        # K[:3, :3] = self.camera._camera_intr._K
        # K[0, 2] = 550 + 128
        # K[1, 2] = 280 + 128
        # K[2, 2] = -1
        x_dim = 1032
        y_dim = 772
        clip_x = 190+130
        clip_y = 190
        height = y_dim - 2*clip_y
        width = x_dim - 2*clip_x
        #Row of crop window center
        crop_ci = clip_y + height / 2
        crop_cj = clip_x + width / 2
        # K = np.identity(4)
        # K[:3, :3] = self.camera._camera_intr._K
        # print("K", K)
        # K[0, 2] -= clip_x
        # K[1, 2] -= clip_y
        # K[0] *= 256/(x_dim - 2*clip_x)
        # # K[1] *= 256 / (x_dim - 2 * clip_x)
        # # K[0] *= 256 / (y_dim - 2 * clip_y)
        # # KRt = K @ Rt
        # K[1] *= 256 / (y_dim - 2 * clip_y)
        # K[2,2] = -1
        camera_intrinsics = self.camera.intrinsics
        # print("K before cropping", self.camera.intrinsics._K)
        camera_intrinsics = camera_intrinsics.crop(height, width, crop_ci, crop_cj)
        camera_intrinsics = camera_intrinsics.resize(256 / height)
        K = np.identity(4)
        K[:3, :3] = camera_intrinsics._K
        # K[2,2] = -1
        # K[:3] *= -1
        print("K", K)

        oprefix = self.filelist[idx]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        # depth_scale = 1 / abs(Rt[2][3])


        result = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float(),
            "K": torch.tensor(K).float()
            # "RT": None,
        }
        #
        # w0, ws = sample_plane(Rt)
        # S = [K @ w2S(w) @ LA.inv(K) for w in ws]
        # y = [to_label(w, w0) for w in ws]
        # result["S"] = None
        # result["y"] = None
        # result["w"] = None
        # result["w0"] = None

        return result

def sample_plane(Rt, plane=np.array([0, 1, 0, 0]), plane2=np.array([1, 0, 0, 0])):
    w0_ = LA.inv(Rt).T @ plane
    # find plane normal s.t. w0 @ x + 1 = 0
    w0 = w0_[:3] / w0_[3]
    # normalize so that w[2]=1
    w0 = w0 / w0[2]

    if CM.detection.sample_hard_negative:
        # sample around second symmetry axis (hard negative)
        w1_ = LA.inv(Rt).T @ plane2
        w1 = w1_[:3] / w1_[3]
        w1 = w1 / w1[2]
        ws = [
            sample_symmetry(w0, 0, math.pi / 2),
            sample_symmetry(w1, 0, CM.detection.theta[0]),
        ]
    else:
        while True:
            w = sample_symmetry(w0, 0, math.pi / 2)
            if sum(to_label(w, w0)) == 0:
                break
        ws = [w]
    for theta1, theta0 in zip(CM.detection.theta, CM.detection.theta[1:] + [0]):
        ws.append(sample_symmetry(w0, theta0 * 1.001, theta1 * 0.999))
    return w0, ws


def sample_symmetry(w0, theta0, theta1, delta=1):
    w = sample_sphere(w0 / LA.norm(w0), theta0, theta1)
    return w / (delta * w[2])


def sample_sphere(v, theta0, theta1):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def w2S(w):
    S = np.eye(4)
    S[:3, :3] = np.eye(3) - 2 * np.outer(w, w) / np.sum(w ** 2)
    S[:3, 3] = -2 * w / np.sum(w ** 2)
    return S


def to_label(w, w0):
    theta = math.acos(np.clip(abs(w @ w0) / LA.norm(w) / LA.norm(w0), -1, 1))
    return [theta < theta0 for theta0 in CM.detection.theta]
