import numpy as np
from scipy import spatial
from sym.datasets import w2S
import open3d as o3d
import pyvista as pv
import trimesh
import os
from autolab_core import ColorImage, Image, PointCloud, RigidTransform
'''
faster mode calculation than scipy.
'''
calib_dir = "/home/lawrence/bilateral_symmetry/calib"

def mode1(x):
    values, counts = np.unique(x, return_counts = True)
    m = counts.argmax()
    return values[m], counts[m]

'''
This function takes as input the normal vector to our symmetry plane, the depth of the segmented out pixel coordinates,
and the camera intrinsics K.
Calculates the reflection plane, the 3D coordinates of these points combined with the reflection completed 3D points.
If we only need to plot 3d points that are in the segmented out area of the image we would need an input of mask.

normal: a 1x3 or 3x1 ndarray w.
depth: a HxW nd array
K: a 3x3 camera intrinsics matrix
mask: a HxW mask representing object segmentation.
filter: whether to filter out abnormal z data that's incorporated into the segmented image but has abnormal depth.
'''
def reflect_2D(normal, depth, K, mask, file_name, filter = True, manual_correct = False, is_real_data = True):
    normal = normal.reshape((3,1))
    normal[2, 0] *= -1
    H, W = depth.shape[0], depth.shape[1]
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    if is_real_data:
        x = x.flatten()
        y = y.flatten()
    else:
        x = x.flatten() * 2 / W - 1
        y = 1 - y.flatten() * 2 / H
    z = depth.flatten()
    xyz = np.vstack([x * z, y * z, z])[:,mask.flatten()]
    if filter:
        mask = xyz[2,:] > 0.45
        xyz = xyz[:, mask]
    XYZ = np.linalg.inv(K) @ xyz
    num_points = XYZ.shape[1]
    order = XYZ[2, :].argsort()
    #We don't want to have points that have too low a z value (very close to the camera and is prone to be wrong to affect our center of object finding.
    XYZ = XYZ[:, order]
    # import pdb; pdb.set_trace()
    if filter:
        center_of_object = np.mean(XYZ[:,int(0.1*num_points):], axis = 1)
    else:
        center_of_object = np.mean(XYZ[:,int(0.1*num_points):int(0.7*num_points)], axis=1)
    center_of_object[1] += 0.075
    print("center_of_object", center_of_object)
    #Now we will construct the mesh.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ.T)
    # #Remove outliers.
    # import pdb;pdb.set_trace()
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    o3d.visualization.draw_geometries([pcd])
    # pcd, ind = pcd.remove_radius_outlier(nb_points = 30, radius = 0.02)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.015, max_nn =30))
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    # poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 9)
    ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.002,0.005,0.01,0.02]))
    #We will remove the low density vertices here.
    # vertices_to_move = densities < np.quantile(densities, 0.05)
    # poisson_mesh.remove_vertices_by_mask(vertices_to_move)
    # o3d.visualization.draw_geometries([poisson_mesh])
    bbox = pcd.get_axis_aligned_bounding_box()
    # # # p_mesh_crop = poisson_mesh.crop(bbox)
    p_mesh_crop = ball_mesh.crop(bbox)
    # print("Cluster connected triangles")
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (
    #         p_mesh_crop.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # print("Show largest cluster")
    # import copy
    # mesh_1 = copy.deepcopy(p_mesh_crop)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    # triangles_to_remove = triangle_clusters != largest_cluster_idx
    # mesh_1.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([ball_mesh])
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=100, print_progress=True))
    max_label, count = mode1(labels)
    print("max label", max_label)
    cluster_mask = (labels == max_label)

    # o3d.visualization.draw_geometries([p_mesh_crop])
    #This is to retrieve the cleaner xyz 3D coordinates that have been filtered by the statistical filter.
    # o3d.visualization.draw_geometries([pcd])
    XYZ = np.asarray(pcd.points)[cluster_mask, :]
    XYZ = XYZ.T
    print("XYZ shape", XYZ.shape)
    if manual_correct:
        corrected_normal = correct_normal(normal, XYZ, center_of_object)
    else:
        corrected_normal = normal
    #Then we calculate the b of w.Tx + b =0
    b = -corrected_normal.T @ center_of_object
    # import pdb;pdb.set_trace()
    #After this, the normal vector is normalized to nx+b
    corrected_normal = corrected_normal / b
    # normal_2 = np.linalg.norm(normal)
    normal_matrix = w2S(corrected_normal.reshape(3,))
    # normal_matrix = np.diag((1.,1.,1.,1.))
    # normal_matrix[:3, 3:] = -2 * normal / (normal_2 ** 2)
    # normal_matrix[:3, :3] = np.identity(3) - 2 / (normal_2 ** 2) * (normal @ normal.T)
    one = np.ones((1, XYZ.shape[1]))
    XYZ_homo = np.concatenate((XYZ, one), axis=0)
    reflected_XYZ_homo = normal_matrix @ XYZ_homo
    reflected_XYZ = reflected_XYZ_homo[:3,:]
    all_XYZ = np.concatenate((XYZ, reflected_XYZ), axis = 1)



    # Transform the object from the camera frame to the robot frame
    phoxi_to_world_fn = os.path.join(
        calib_dir, "phoxi", "phoxi_to_world_etch.tf")
    T_phoxi_world = RigidTransform.load(phoxi_to_world_fn)
    all_XYZ_world_homo = T_phoxi_world.matrix @ np.concatenate((XYZ_homo, reflected_XYZ_homo), axis = 1)
    all_XYZ_world = all_XYZ_world_homo[:-1, :]

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(all_XYZ_world.T)
    # pcd_new, _ = pcd_new.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
    pcd_new.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.015, max_nn =30))
    pcd_new.orient_normals_towards_camera_location(pcd_new.get_center())
    pcd_new.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_new.normals))
    distances = pcd_new.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist  
    # poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 9)
    ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_new, o3d.utility.DoubleVector([radius, radius * 2, radius * 4]))
    #We will remove the low density vertices here.
    # vertices_to_move = densities < np.quantile(densities, 0.05)
    # poisson_mesh.remove_vertices_by_mask(vertices_to_move)
    # o3d.visualization.draw_geometries([poisson_mesh])
    o3d.visualization.draw_geometries([pcd_new])
    bbox = pcd_new.get_axis_aligned_bounding_box()
    # # p_mesh_crop = poisson_mesh.crop(bbox)
    p_mesh_crop = ball_mesh.crop(bbox)
    o3d.visualization.draw_geometries([p_mesh_crop])
    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(p_mesh_crop.vertices), np.asarray(p_mesh_crop.triangles),
                          vertex_normals=np.asarray(p_mesh_crop.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)
    #Saving meshes here.
    # o3d.io.write_triangle_mesh("./data/real_data2/test_car.stl", p_mesh_crop)
    trimesh.exchange.export.export_mesh(tri_mesh, './data/real_data2/' + 'test' + file_name +'.obj')
    # print("Cluster connected triangles")
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (
    #         p_mesh_crop.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area)
    # print("Show largest cluster")
    # import copy
    # mesh_1 = copy.deepcopy(p_mesh_crop)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    # triangles_to_remove = triangle_clusters != largest_cluster_idx
    # mesh_1.remove_triangles_by_mask(triangles_to_remove)
    # o3d.visualization.draw_geometries([mesh_1])

    # downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    # downsampled_point_cloud = np.asarray(downpcd.points)
    # cloud = pv.PolyData(downsampled_point_cloud).extract_geometry().triangulate()
    # cloud = pv.PolyData(XYZ.T).extract_geometry().triangulate()

    # cloud = pv.PolyData(all_XYZ.T).extract_geometry().triangulate()
    # volume = cloud.delaunay_3d(alpha=.01)
    # volume.plot(show_edges = True)
    return corrected_normal, all_XYZ, center_of_object

'''
A function for correcting the outputted z from NeRD. It samples z from [-1,1] to find the one that works the best for the given normal.
normal: 3x1 ndarray that specifies the normal direction that we will use as a template direction.
XYZ: 3xn ndarray. It's the pointcloud of the object that we're interested in finding its symmetry.
center_of_object: 3x1 ndarray, used to calculate where the symmetry plane should lie on.
'''
def correct_normal(normal, XYZ, center_of_object):
    normal_normalized = normal / np.linalg.norm(normal)
    #We know that nerd outputted z is always 1 and the xy coordinates are usually much bigger than z coordinates
    #Setting that z can vary in between -0.8 and +0.8 is a good assumption.
    # z_list = np.linspace(-0.8, -0.55, 11)
    z_list = np.linspace(-0.8, -normal_normalized[2] - 0.2, 21)
    # z_list = [normal_normalized[2]]
    # z_list = [-0.4]
    # y_abs = abs(normal[1])
    y_list = np.linspace(normal_normalized[1] - 0.3, normal_normalized[1] + 0.3, 11)
    num_pts = XYZ.shape[1]
    dist_list = []
    XYZ_reformat = XYZ.reshape((-1,3))
    XYZ_kdtree = spatial.KDTree(XYZ_reformat)
    dist_array = np.ones((len(z_list), len(y_list)))
    for indice in range(len(z_list)):
        z = z_list[indice]
        for y_index in range(len(y_list)):
            y = y_list[y_index]
            total_dist = []
            normal_test = normal_normalized.copy()
            normal_test[-1] = z
            normal_test[-2] = y
            b = -normal_test.T @ center_of_object
            normal_test = normal_test / b
            normal_matrix = w2S(normal_test.reshape(3, ))
            one = np.ones((1, XYZ.shape[1]))
            XYZ_homo = np.concatenate((XYZ, one), axis=0)
            # import pdb;pdb.set_trace()
            reflected_XYZ_homo = normal_matrix @ XYZ_homo
            reflected_XYZ = reflected_XYZ_homo[:3, :].reshape((-1,3))
            # XYZ_reformat = XYZ.reshape((-1,3))
            # reflected_kdtree = spatial.KDTree(reflected_XYZ)
            for index in range(num_pts):
                pt = reflected_XYZ[index, :]
                distance, _ = XYZ_kdtree.query(pt)
                total_dist.append(distance)
            total_dist = np.sort(total_dist)[int(0.*num_pts):]
            dist_array[indice][y_index] = np.mean(total_dist)
            print("finished round " + str(indice) + str(y_index))
    z_indice, y_indice = np.argwhere(dist_array == np.min(dist_array))[0]
    best_normal = normal_normalized.copy()
    # best_normal[-1] = z_list[best_ind]
    best_normal[-1] = z_list[z_indice]
    best_normal[-2] = y_list[y_indice]
    print("best_normal", best_normal)
    return best_normal












