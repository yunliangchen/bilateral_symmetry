#!/usr/bin/env python3
import os
import os.path as osp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import numpy.linalg as LA
import cv2
from autolab_core import ColorImage, Image, PointCloud, RigidTransform
from perception.webcam_sensor import WebcamSensor
import numpy as np
from phoxipy import PhoXiSensor
from perception.colorized_phoxi_sensor import ColorizedPhoXiSensor
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.instance import instance_segmentation
from perception.webcam_sensor import WebcamSensor
import numpy as np
import cv2
# from .points import Point, PointCloud, ImageCoords

calib_dir = "/home/lawrence/bilateral_symmetry/calib"
phoxi_config = dict(frame = "phoxi", device_name = "1703005", size = "small")
webcam_config = dict(frame = "webcam", device_id = 0)
p = PhoXiSensor("1703005")
p.start()
p._set_intr()
frame, normals = p.read()
depth_image = frame.depth
depth_image._frame = "phoxi"
# webcam = WebcamSensor()
# webcam.start()
# color_image, _ = webcam.frames()
input_image_dir = "./data/real_data2/pawn.png"
initial_image_dir = "./data/real_data2/capture.jpg"
output_image_dir = "./data/real_data2/pawn_cropped_segmented.npy"
color_img = cv2.imread(initial_image_dir, cv2.IMREAD_UNCHANGED)
print(color_img.shape)
plt.imshow(color_img)
plt.show()
color_image = ColorImage(color_img, frame = "webcam", encoding = 'rgb8')
colorize_phoxi_sensor = ColorizedPhoXiSensor(phoxi_config, webcam_config, calib_dir, frame=depth_image._frame)
K = p._intrinsics._K
phoxi_to_world_fn = os.path.join(
    calib_dir, "phoxi", "phoxi_to_world_etch.tf")
T_phoxi_world = RigidTransform.load(phoxi_to_world_fn)
plt.imshow(depth_image.data)
plt.show()
color_depth_im = colorize_phoxi_sensor._colorize(depth_image, color_image)
print("colorized_depth_image shape", color_depth_im.data.shape)
plt.imshow(color_depth_im.data)
plt.show()

def transform_3d(pixel_coord, K, phoxi_world, depth_image):
    #This pixel coord means what's the coordinate if we view the horizontal axis as x and vertial ones as y.
    #The reason we are changing x and y is because np array is indexed by row then column but image's x is column first.
    x = pixel_coord[0]
    y = pixel_coord[1]
    depth = depth_image[y, x]
    homo_2d = depth * np.array([x, y, 1])
    point3d = np.linalg.inv(K) @ homo_2d
    point3d_homog = np.r_[point3d, 1]
    transformed = phoxi_world @ (point3d_homog)
    return transformed

print("depth_image shape", depth_image.shape)

clip_y = 190
clip_x = clip_y + 130
y_dim = color_depth_im.data.shape[0]

x_dim = color_depth_im.data.shape[1]
print("dimensions of phoxi sensor", y_dim, x_dim)
height = y_dim - 2*clip_y
width = x_dim - 2*clip_x
rgb_data = color_depth_im.data[clip_y:y_dim - clip_y, clip_x:x_dim - clip_x, :]

depth_data = depth_image.data[clip_y:y_dim - clip_y, clip_x:x_dim - clip_x].reshape(height, width, 1)
min_depth = np.min(depth_data[np.nonzero(depth_data)])
print(min_depth)

rgbd_data = np.concatenate((rgb_data, depth_data), axis = -1)
print("cropped rgb image shape:", rgb_data.shape)
small = cv2.resize(rgbd_data, (256, 256))
small[small < min_depth] = 0
rgb_data = small[:,:, :3]
depth_data = small[:, :, 3:4]
# import pdb;pdb.set_trace()

#Now we need to do color thresholding to remove the pure black pixel values which represent pixels with zero depths in images.
lower = np.array([1,1,1])
upper = np.array([255,255,255])
mask = cv2.inRange(rgb_data, lower, upper)
# print(mask)
black_mask = (mask!=0)
# print(rgb_data.shape)
cv2.imwrite(input_image_dir, rgb_data)
print("saved image")

# webcam.stop()
# input_image_dir = "./data/real_data2/pawn_cropped.png"
ins = instanceSegmentation()
ins.load_model("./pointrend_resnet50.pkl")
# ins.load_model("./pointrend_resnet50.pkl", bboxes = True)
segmask, output = ins.segmentImage(input_image_dir, output_image_name="./data/real_data2/pawn_pixellib.png", show_bboxes=True)
# segment_image = instance_segmentation()
# segment_image.load_model("./mask_rcnn_coco.h5")
# segmask, oputput = segment_image.segmentImage(input_image_dir, outpout_image_name = "./data/real_data2/test_pixellib.png", show_bboxes=True)
print("finished with the segmask")
# print("segmask", segmask)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# print(segmask)
# import pdb; pdb.set_trace()
scores = segmask['scores']
segmask = segmask['masks']

assert segmask.shape[2] > 0, f"pixellib does not detect any object, expecting greater than 0 but got: {segmask.shape[2]}"
if segmask.shape[2] > 1:
    print("There are two or more objects!\n")
    indice = np.argsort(scores)
    biggest_index = indice[0]
    # print("scores are", scores.cpu().numpy())
    print("The score of the object chosen is ", scores[biggest_index].cpu().numpy())
    segmask = segmask[:,:,biggest_index]
# import pdb;pdb.set_trace()
black_mask = black_mask.reshape(segmask.shape)
segmask = np.logical_and(segmask, black_mask)

print("finished the logical and between segmask and the black depth mask")
# print("segmask shape", segmask.shape)
# import pdb; pdb.set_trace()
# image = cv2.imread(input_image_dir)
# cv2.imshow(image)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# alpha_channel = image[:, :, 3]
# alpha_channel = alpha_channel.reshape((image.shape[0], image.shape[1], 1))
# # print(segmask.shape)
# # print(image.shape)
# # segmask = segmask.reshape((segmask.shape[1], segmask.shape[2], 1))
# alpha_channel_template = np.zeros(alpha_channel.shape)
# alpha_channel_template[segmask] = alpha_channel[segmask] *255
image = small
depth_channel = np.zeros(depth_data.shape)
depth_channel[segmask] = depth_data[segmask]

# import pdb; pdb.set_trace()
# image[:, :, 3] = alpha_channel_template.reshape((alpha_channel_template.shape[0], alpha_channel_template.shape[1]))
# image[:, :, 3:] = alpha_channel_template
image[:,:,3:] = depth_channel
# plt.imshow(image[:,:, :3])
# plt.show()
# import pdb;pdb.set_trace()
# cv2.imwrite(output_image_dir, image)
with open(output_image_dir, 'wb') as f:
    np.save(f, image)
# import pdb;pdb.set_trace()

# image = np.concatenate((image, depth_data), axis = -1)
#Now begin the second stage, changing K and feeding it into the real evaluation module.
# image = cv2.imread(output_image_dir, cv2.IMREAD_UNCHANGED)

camera_intr = p._intrinsics
# print("K", K)
# K[0, 2] -= clip_x
# K[1, 2] -= clip_y
# # K[0] *= 256/(x_dim - 2*clip_x)
# K[1] *= 256/(y_dim - 2*clip_y)
# K[0] *= 256/(x_dim - 2*clip_x)
# K[1] *= 256/(y_dim - 2*clip_y)
# x_dim = 1280
# y_dim = 960
#Row of crop window center
crop_ci = clip_y + height / 2
crop_cj = clip_x + width / 2
camera_intrinsics = camera_intr.crop(height, width, crop_ci, crop_cj)
camera_intrinsics = camera_intrinsics.resize(256 / height)
# pixel = np.meshgrid()
K = camera_intrinsics._K
# real_world_coordinate = transform_3d([128,40], K, T_phoxi_world.matrix, image[:, :, 3])
# print("\nreal world 3d coordinate", real_world_coordinate)

# print("recovered 3D point", )
# print("K_modified", K)


# print("outputted image shape", image.shape)
p.stop()
# webcam.stop()

