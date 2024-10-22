from lib import keypoint_features, io, normal

'''
事前に以下を実施すること
$ cd 3rdparty/Open3D/examples/test_data/
$ wget http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
$ tar xvfz bunny.tar.gz
$ mv bunny/reconstruction/bun_zipper.ply Bunny.ply
$ rm -r {bunny,bunny.tar.gz}
'''

file_bunny = '3rdparty/Open3D/examples/test_data/Bunny.ply'

# chap3.keypoint features #########################
# 共通 ---------
pcd = io.load_point_cloud(file_bunny)
# 法線推定
pcd_normal = normal.estimate_vertex_normals(pcd)
pcd_normal.paint_uniform_color([0.5, 0.5, 0.5])

# Harris3D -----
print('KeyPoints: Harris3D')
kpts_h3d = keypoint_features.harris3d_keypoints(pcd)
io.draw_simple([keypoint_features.keypoints_to_spheres(kpts_h3d), pcd_normal], "特徴点 Harris3D")


# ISS ----------
print('KeyPoints: ISS')
kpts_iss = keypoint_features.iss_keypoint(pcd)
io.draw_simple([keypoint_features.keypoints_to_spheres(kpts_iss), pcd_normal], "特徴点 ISS")

# 局所特徴量 FPFH　----------
print('features: FPFH')
fpfh = keypoint_features.fpfh_features(pcd)
