import numpy as np
import copy

import open3d as o3d
from lib import io, recognition
import matplotlib.pyplot as plt

# 平面
def ransac_plane():
    zoom=0.459
    front=[ 0.349, 0.063, -0.934 ]
    lookat=[ 0.039, 0.007, 0.524 ]
    up=[ -0.316, -0.930, -0.181 ]
    pcd=io.load_point_cloud("./data/tabletop_scene.ply")
    io.draw_detail([pcd], zoom, front, lookat, up, window_name='ransac_plane: ref')
    # 平面セグメント
    plane_model, inliers = recognition.segment_plane(pcd)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # 平面の点群データ
    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color([1.0, 0, 0])
    # 平面以外の点群データ
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # 描画
    io.draw_detail([plane_cloud, outlier_cloud], zoom, front, lookat, up, window_name='ransac_plane: plane')
    io.draw_detail([outlier_cloud], zoom, front, lookat, up, window_name='ransac_plane: outlier')
    
# 球体の検出
def ransac_sphere():
    zoom=0.459
    front=[ 0.349, 0.063, -0.934 ]
    lookat=[ 0.039, 0.007, 0.524 ]
    up=[ -0.316, -0.930, -0.181 ]
    # データ読み込み
    pcd = io.load_point_cloud("./data/tabletop_scene_segment.ply") # planeのoutlierと同じ点群データ
    np_pcd = np.asarray(pcd.points)
    io.draw_detail([pcd], zoom, front, lookat, up, window_name='ransac_sphere: ref')

    # Parameters
    ransac_n = 4 # 点群から選択する点数．球の場合は4．
    num_iterations = 1000 # RANSACの試行回数
    distance_th = 0.005 # モデルと点群の距離のしきい値
    max_radius = 0.05 # 検出する球の半径の最大値

    # 解の初期化
    best_fitness = 0 # モデルの当てはめの良さ．インライア点数/全点数
    best_inlier_dist = 10000.0 #インライア点の平均距離
    best_inliers = None # 元の点群におけるインライアのインデクス
    best_coeff = np.zeros(4) # モデルパラメータ

    for n in range(num_iterations):
        c_id = np.random.choice( np_pcd.shape[0], 4, replace=False )
        coeff = recognition.ComputeSphereCoefficient( np_pcd[c_id[0]], np_pcd[c_id[1]], np_pcd[c_id[2]], np_pcd[c_id[3]] )
        if max_radius < coeff[3]:
            continue
        fitness, inlier_dist, inliers = recognition.EvaluateSphereCoefficient( np_pcd, coeff, distance_th )
        if (best_fitness < fitness) or ((best_fitness == fitness) and (inlier_dist<best_inlier_dist)):
            best_fitness = fitness
            best_inlier_dist = inlier_dist
            best_inliers = inliers
            best_coeff = coeff
            print(f"Update: Fitness = {best_fitness:.4f}, Inlier_dist = {best_inlier_dist:.4f}")

    if best_coeff.any() != False:
        print(f"Sphere equation: (x-{best_coeff[0]:.2f})^2 + (y-{best_coeff[1]:.2f})^2 + (z-{best_coeff[2]:.2f})^2 = {best_coeff[3]:.2f}^2")
    else:
        print("No sphere detected.")

    # 結果の可視化
    sphere_cloud = pcd.select_by_index(best_inliers)
    sphere_cloud.paint_uniform_color([0, 0, 1.0])
    outlier_cloud = pcd.select_by_index(best_inliers, invert=True)

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=best_coeff[3])
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.3, 0.3, 0.7])
    mesh_sphere.translate(best_coeff[:3])
    o3d.visualization.draw_geometries([mesh_sphere]+[sphere_cloud+outlier_cloud])

    # 球の点群データ
    plane_cloud = pcd.select_by_index(best_inliers)
    plane_cloud.paint_uniform_color([0, 0, 1.0])
    # 平面以外の点群データ
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # 描画
    io.draw_detail([plane_cloud, outlier_cloud], zoom, front, lookat, up, window_name='ransac_sphere: sphere')

def segmentation():
    filename = "./data/tabletop_scene_segment.ply"
    # filename = "./3rdparty/Open3D/examples/test_data/fragment.pcd"
    print("Loading a point cloud from", filename)
    pcd = o3d.io.read_point_cloud(filename)
    print(pcd)

    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / max(max_label,1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd], zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

# ransac_plane()
# ransac_sphere()
segmentation()
