#!/usr/bin/env python3
"""Compute vanishing points using corase-to-fine method on the evaluation dataset.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --visualize <outdir>          Output visualization related files
   --suffix <suffix>             File suffix of visualization [default: nerd]
   --split <split>               Split for testing [default: test_all]
   --noimshow                    Do not show result
"""

import math
import os
import os.path as osp
import pprint
import random
import shlex
import subprocess
import sys
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import numpy.linalg as LA
import skimage.io
import torch
from docopt import docopt
from tqdm import tqdm

import sym
from sym.config import CI, CM, CO, C
from sym.datasets import Pix3dDataset, ShapeNetDataset, to_label, w2S, RealDataset
from sym.models import SymmetryNet
from sym.utils import np_eigen_scale_invariant, np_kitti_error
from reflect_and_mesh import reflect_2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold

def w2S(w):
    S = np.eye(4)
    S[:3, :3] = np.eye(3) - 2 * np.outer(w, w) / np.sum(w ** 2)
    S[:3, 3] = -2 * w / np.sum(w ** 2)
    return S

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    CI.update(C.io)
    CM.update(C.model)
    CO.update(C.optim)
    pprint.pprint(C, indent=4)

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    print(device)
    print("Working on", args["<checkpoint>"])
    checkpoint = torch.load(args["<checkpoint>"])
    model = sym.models.SymmetryNet().to(device)
    model = sym.utils.MyDataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    missing, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert len(missing) == 0
    model.eval()

    if CI.dataset == "ShapeNet":
        Dataset = ShapeNetDataset
    elif CI.dataset == "Pix3D":
        Dataset = Pix3dDataset
    elif CI.dataset == "RealData":
        Dataset = RealDataset
    else:
        raise NotImplementedError
    split = args["--split"]
    # split = "test_all"
    # if "only_car_plane_chair" in CI and CI.only_car_plane_chair:
    #     split = "test_unseen_all"
    # split = "test-1000"
    # if "only_car_plane_chair" in CI and CI.only_car_plane_chair:
    #     split = "test_unseen-1000"

    loader = torch.utils.data.DataLoader(
        Dataset(C.io.datadir, split=split),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    fpath = args["--output"]
    print("save to", fpath)

    thetas = [np.pi / 2] + CM.detection.theta

    err_normal = []
    err_depth_avg = []
    err_depth_SIL = []
    err_depth_AIO = []
    w_pd = []
    w_gt = []
    # D_pd = []
    # D_gt = []

    loader_tqdm = tqdm(loader)
    for batch_idx, input in enumerate(loader_tqdm):
        # import pdb; pdb.set_trace()
        image_name = input["fname"].cpu().numpy()[0]
        image_name = ''.join([chr(i) for i in image_name if i != 0])
        image_name = image_name.split("/")[-1]
        image_name.strip(".npy\x00")
        score = []
        depth = []
        image = input["image"].cpu().numpy()[0][0]
        H, W = image.shape
        print(H, W)
        if CI.dataset != "RealData":
            depth_gt = input["depth"].cpu().numpy()[0][0]
            # import pdb;pdb.set_trace()
            mask = (depth_gt > 0).reshape((H, W))
            print((depth_gt > 0).shape)
            print(mask)
            w0 = input["w0"].cpu().numpy()[0]
            print("Ground truth w:", w0)
        else:
            # import pdb;pdb.set_trace()
            depth_gt = input["depth"].cpu().numpy()[0].reshape((H,W))
            mask = (depth_gt > 0).reshape((H, W))

        # depth_gt = input["depth"].cpu().numpy()[0][0]
        # H, W = depth_gt.shape
        # Rt = input["RT"].cpu().numpy()[0]
        w = np.array([0, 0, 1])
        ww = []

        # print(input["fname"][0].cpu().numpy().tostring().decode("ascii"))

        for i in range(CM.detection.n_level):
            ws, S = sample_reflection(input, w, thetas[i])
            with torch.no_grad():
                input["S"] = torch.tensor(S[None]).float()
                input["w"] = torch.tensor(ws).float()[None]
                # print(input["S"])
                # print(input["w"])
                result = model(input, "test", real=True)
            score = result["preds"]["score"].cpu().numpy()[0, :, i]
            print(result["preds"]["score"].shape)
            print("Score", score)
            depth = result["preds"]["depth"].cpu().numpy()[0, :]
            del result
            # wc,scorec = ws.copy(), score.copy()
            
            # ws /= LA.norm(ws, axis=1, keepdims=True) / 1.02
            # visualize(ws, score, w0=None)
            best_w_index = np.argmax(score)
            # print(np.array_equal(wc, ws))
            # print(np.array_equal(score, scorec))
            w = ws[best_w_index]
            print(w)
            ww.append(w)
        print(w)
        # rescale depth according to the ||w||_2
        depth_pd = depth[best_w_index] #* abs(Rt[2][3])
        with open(f'temp_output/{CI.dataset}/depth_map_{batch_idx}.npy', 'wb') as f:
            np.save(f, depth_pd)

        print(depth_pd.shape)
        # import pdb; pdb.set_trace()

        use_depth = False
        if use_depth:
            folder_name = f'{CI.dataset}_with_depth'
        else:
            folder_name = f'{CI.dataset}'

        plt.figure()
        if CI.dataset != "RealData":
            plt.imshow(depth_pd * mask, cmap='bwr')
        else:
            plt.imshow(depth_pd * mask, norm=plt.Normalize(0.8, 1.2), cmap='bwr')
        plt.colorbar()
        plt.savefig(f'temp_output/{folder_name}/depth_map_{batch_idx}.png')
        # plt.show()
        plt.clf()

        plt.figure()
        plt.imshow(image)
        plt.savefig(f'temp_output/{folder_name}/input_{batch_idx}.png')
        # plt.show()
        plt.clf()
        # import pdb;pdb.set_trace()
        plt.figure()
        if CI.dataset != "RealData":
            plt.imshow(depth_gt * mask, cmap='bwr')
        else:
            plt.imshow(depth_gt * mask, norm=plt.Normalize(0.5, 0.9), cmap='bwr')
        plt.colorbar()
        plt.savefig(f'temp_output/{folder_name}/depth_map_gt_{batch_idx}.png')
        # plt.show()
        plt.clf()


        
        # This is specifically the drawing method for synthetic dataset.
        if CI.dataset != "RealData":
            x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
            # y, x = np.meshgrid(np.arange(0, W), np.arange(0, H))
            x = x.flatten() * 2 / W - 1
            y = 1 - y.flatten() * 2 / H
            z = depth_gt.flatten()
            print(input["K"])
            K = input["K"][0, :3, :3]
        # This is specifically the drawing method for real world dataset where we don't need to normalize the x and y.
        else:
            x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
            # y, x = np.meshgrid(np.arange(0, W), np.arange(0, H))
            x = x.flatten()
            # y = H - y.flatten()
            y = y.flatten()
            # z = depth_pd.flatten()
            z = depth_gt.flatten()
            print(input["K"])
            K = input["K"][0, :3, :3]

            # K[2,2] = -K[2,2] # the input K is negative. Here we make it positive.

        if CI.dataset != "RealData":
            xyz = np.vstack([x * z, y * z, z])[:, mask.flatten()]
        else:
            xyz = np.vstack([x * z, y * z, z])[:, mask.flatten()]
        # mask = xyz[2,:] > 0.2
        # xyz = xyz[:,mask]
        
        XYZ = np.linalg.inv(K) @ xyz
        with open(f'temp_output/{folder_name}/reconstructed_3d_{batch_idx}.npy', 'wb') as f:
            np.save(f, XYZ)
        center_of_object = np.mean(XYZ, axis=1)
        print(center_of_object)

        # ax = plt.axes(projection='3d')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect(aspect = [1, 1, 1])
        # ax.scatter3D(XYZ[0, :], XYZ[1, :], symmetry_z)
        # scatter_plot = ax.scatter3D(XYZ[0, :], XYZ[1, :], XYZ[2, :], c=XYZ[2, :], norm=plt.Normalize(.4,  1.3), cmap='Greens')
        
        # homo_xyz = np.vstack([XYZ, np.ones((1, xyz.shape[1]))])
        # reflected_xyz = w2S(w) @ homo_xyz
        # reflected_xyz = reflected_xyz / reflected_xyz[-1, :]
        # ax.scatter3D(reflected_xyz[0, :], reflected_xyz[1, :], reflected_xyz[2, :], c = reflected_xyz[2, :], cmap='Reds')
        # mesh_x, mesh_y = np.meshgrid(np.linspace(np.min(XYZ[0, :]), np.max(XYZ[0, :]), 50),
        #                              np.linspace(np.min(XYZ[1, :]), np.max(XYZ[1, :]), 50))
        # symmetry_z = (-1 - w[0] * mesh_x - w[1] * mesh_y) / w[2]
        # w = w / np.linalg.norm(w) / 6
        # ax.plot3D([center_of_object[0], center_of_object[0]+w[0]], [center_of_object[1], center_of_object[1]+w[1]], [center_of_object[2], center_of_object[2]+w[2]], 'red', linewidth = 10.0)

        w_correct, all_XYZ, center_of_object = reflect_2D(w, depth_gt, K, mask, image_name, filter=False, manual_correct=True, is_real_data=True)
        # w_correct, all_XYZ, center_of_object = reflect_2D(w, depth_gt, K, mask, image_name, filter=False, manual_correct=True, is_real_data=True)
        # scatter_plot = ax.scatter3D(all_XYZ[0, :], all_XYZ[1, :], all_XYZ[2, :], c=all_XYZ[2, :], cmap='Reds')

        mesh_x, mesh_y = np.meshgrid(np.linspace(np.min(XYZ[0, :]), np.max(XYZ[0, :]), 50),
                                     np.linspace(np.min(XYZ[1, :]), np.max(XYZ[1, :]), 50))
        # print(mesh_x, mesh_y)
        symmetry_z = (-1 - w_correct[0] * mesh_x - w_correct[1] * mesh_y) / w_correct[2]
        # print(symmetry_z)
        # ax.plot_wireframe(mesh_x, mesh_y, symmetry_z, rstride=1, cstride=1)
        ax.plot_surface(mesh_x, mesh_y, symmetry_z)
        ax.scatter3D(center_of_object[0], center_of_object[1], center_of_object[2], s=15**2, c='k')
        scatter_plot = ax.scatter3D(XYZ[0, :], XYZ[1, :], XYZ[2, :], c=XYZ[2, :], cmap='Greens')

        # ax.plot3D([0,w[0]], [0,w[1]], [0,w[2]], 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_zlim([0.6,0.9])
        cb = plt.colorbar(scatter_plot)
        # ax.view_init(elev=-90, azim=-90)
        ax.view_init(elev=90, azim=-90)
        plt.savefig(f'temp_output/{folder_name}/reconstructed_3d_{batch_idx}.png')
        plt.show()
        plt.clf()

        # depth_gt = depth_gt * abs(Rt[2][3])
        # w /= abs(Rt[2][3])
        # w0 /= abs(Rt[2][3])

        # mask = np.logical_and(depth_gt > 0, depth_pd > 0)
        # diff = np.abs(depth_gt - depth_pd)[mask]

        if args["--visualize"]:
            fname = input["fname"].cpu().numpy()[0].tostring().decode("ascii")
            fname = fname.rstrip("\x00")
            if CI.dataset == "ShapeNet":
                fname = fname[::-1].replace("/", "_", 1)[::-1]

            fname_pd = f"{args['--visualize']}/{fname}_{args['--suffix']}.npz"
            fname_gt = f"{args['--visualize']}/{fname}_gt.npz"
            os.makedirs(osp.dirname(fname_pd), exist_ok=True)
            # np.savez(fname_pd, w=w, ww=np.array(ww))
            # np.savez(fname_gt, w=w0)
            # np.savez(fname_pd, depth=depth_pd, w=w)
            # np.savez(fname_gt, depth=depth_gt, w=w0)

        # err_depth_avg += [np.average(diff)]
        # err_depth_SIL += [np_eigen_scale_invariant(depth_pd, depth_gt, mask)]
        # err_depth_AIO.append(np_kitti_error(depth_gt, depth_pd, mask))
        # err_normal += [np.arccos(min(1, abs(w @ w0 / LA.norm(w0) / LA.norm(w))))]
        # w_pd += [w]
        # w_gt += [w0]

    # err_normal = np.sort(np.array(err_normal)) / np.pi * 180
    # err_depth_avg = np.sort(err_depth_avg)
    # err_depth_SIL = np.sort(err_depth_SIL)
    # err_depth_AIO = np.average(err_depth_AIO, axis=0)
    #
    # print("avg:", np.average(err_normal))
    # print("med:", err_normal[len(err_normal) // 2])
    # print("<0d: ", np.sum(err_normal < 0.5) / len(err_normal))
    # print("<1d: ", np.sum(err_normal < 1) / len(err_normal))
    # print("<2d: ", np.sum(err_normal < 2) / len(err_normal))
    # print("<5d: ", np.sum(err_normal < 5) / len(err_normal))
    # print(
    #     "AIO: ",
    #     np.sum(err_normal < 0.5) / len(err_normal),
    #     np.sum(err_normal < 1) / len(err_normal),
    #     np.sum(err_normal < 2) / len(err_normal),
    #     np.sum(err_normal < 5) / len(err_normal),
    # )
    #
    # labels = ["abs_rel", "sq_rel", "rmse", "rmse_log", "sil", "a1", "a2", "a3"]
    # for i, name in enumerate(labels):
    #     print(f"{name}:", err_depth_AIO[i])
    #
    # np.savez(
    #     fpath,
    #     err_depth_avg=err_depth_avg,
    #     err_depth_SIL=err_depth_SIL,
    #     err_depth_AIO=err_depth_AIO,
    #     err_normal=err_normal,
    #     w_pd=np.array(w_pd),
    #     w_gt=np.array(w_gt),
    #     # D_pd=np.array(D_pd),
    #     # D_gt=np.array(D_gt),
    # )

    # if not args["--noimshow"]:
    #     y = (1 + np.arange(len(err_normal))) / len(err_normal)
    #     plt.figure()
    #     plt.plot(err_normal, y)
    #     plt.xlim([0, 5])
    #     plt.grid()
    #     plt.title("normal error")
    #     plt.legend()
    #     plt.show()


def sample_sphere(v, alpha, num_pts):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    w = (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T
    return w


def sample_reflection(input, v, alpha):
    K = input["K"].cpu().numpy()[0]
    ws = sample_sphere(v / LA.norm(v), alpha, CM.detection.n_theta)
    ws /= ws[:, 2:]
    Ss = np.array([K @ w2S(w) @ LA.inv(K) for w in ws])
    return ws, Ss


# count = 0


def visualize(ws, score, w0):
    # w0 /= LA.norm(w0) / 1.02
    # ws /= LA.norm(ws, axis=1, keepdims=True) / 1.02
    ax = plt.figure(figsize=(10, 6)).add_subplot(111, projection="3d")

    # ax.set_box_aspect((1, 1, 1))
    # ax.set_box_aspect((1, 1, 1))
    ax.view_init(27, -22)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    # draw a hemisphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="g", alpha=0.3)

    # draw sampled points
    _ = ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2], c=score)
    # ax.scatter(w0[0], w0[1], w0[2], c="red", marker="^")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0, 1)
    # plt.colorbar(cb)
    # global count
    # ax.set_title(f"Coarse-to-fine Inference Round {count+1}", pad=10)
    # plt.savefig(f"{count}.pdf")
    # count += 1
    plt.show()


if __name__ == "__main__":
    main()
