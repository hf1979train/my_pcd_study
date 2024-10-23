import numpy as np

from lib import basis, io, point_sampling, outlier, normal

file1_bathtub = '3rdparty/Open3D/examples/test_data/bathtub_0154.ply'
file_fragment = '3rdparty/Open3D/examples/test_data/fragment.ply'
file_knot = '3rdparty/Open3D/examples/test_data/knot.ply'

def pcd_calculation(pcd):
    # 最初の描画
    io.draw_simple([pcd], '最初の描画')
    print("回転")
    pcd_r = basis.rotate(pcd, np.pi/3, 0, 0)
    io.draw_simple([pcd_r],'回転')
    print("並進")
    pcd_t = basis.translate(pcd, 1, 0, 0)
    io.draw_simple([pcd_t], '並進')
    print("スケーリング")
    pcd_s = basis.scaling(pcd, 0.5)
    io.draw_simple([pcd_s],'スケーリング')
    print("等間隔サンプリング: ボクセル化")
    io.draw_detail([pcd], *io.get_draw_parameter(), '等間隔サンプリング: ref')
    downpcd_voxel = point_sampling.boxelize_point(pcd, 0.03)
    io.draw_detail([downpcd_voxel], *io.get_draw_parameter(), '等間隔サンプリング:voxel化')
    print("等間隔サンプリング: FPS")
    downpcd_fps = point_sampling.fps_point(pcd, 1000)
    io.draw_detail([downpcd_fps], *io.get_draw_parameter(), '等間隔サンプリング: FPS')
    print("外れ値除去: 統計的")
    outl_st= outlier.remove_statistical_outlier(pcd)
    io.draw_inlier_outlier(pcd, outl_st, *io.get_draw_parameter(), "外れ値除去: 統計的")
    print("外れ値除去: 半径")
    outl_rad= outlier.remove_statistical_outlier(pcd)
    io.draw_inlier_outlier(pcd, outl_rad, *io.get_draw_parameter(), "外れ値除去: 半径")
    print("法線推定(pcd)")
    io.draw_simple([pcd], '法線推定: ref')
    pcd_normal = normal.estimate_vertex_normals(pcd)
    io.draw_simple([pcd_normal], '法線推定: 処置後')
    io.draw_simple([pcd_normal], '法線推定: 処置後', point_show_normal=True)


def mesh_calculation(mesh):
    # 最初の描画
    io.draw_detail([mesh], *io.get_draw_parameter_mesh(), '最初の描画(wireframeなし)')
    io.draw_detail_mesh([mesh], *io.get_draw_parameter_mesh(), '最初の描画(wireframeあり)')
    print("回転")
    mesh_r = basis.rotate(mesh, np.pi/3, 0, 0)
    io.draw_detail_mesh([mesh, mesh_r], *io.get_draw_parameter_mesh(), '回転')
    print("並進")
    mesh_t = basis.translate(mesh, 1, 0, 0)
    io.draw_detail_mesh([mesh, mesh_t], *io.get_draw_parameter_mesh(), '並進')
    print("スケーリング")
    mesh_s = basis.scaling(mesh, 0.5)
    io.draw_detail_mesh([mesh, mesh_s], *io.get_draw_parameter_mesh(), 'スケーリング')
    print("等間隔サンプリング: ボクセル化")
    io.draw_detail_mesh([mesh], *io.get_draw_parameter_mesh(), '等間隔サンプリング: ref')
    pcd_mesh = io.convert_mesh2pcd(mesh)
    downpcd_voxel = point_sampling.boxelize_point(pcd_mesh, 0.03)
    io.draw_detail([downpcd_voxel], *io.get_draw_parameter_mesh(), '等間隔サンプリング:voxel化')
    print("等間隔サンプリング: FPS")
    downpcd_fps = point_sampling.fps_point(pcd_mesh, 1000)
    io.draw_detail([downpcd_fps], *io.get_draw_parameter_mesh(), '等間隔サンプリング: FPS')
    print("等間隔サンプリング: POS")
    downpcd_pos = point_sampling.pos_point_mesh(mesh, 500)
    io.draw_detail([downpcd_pos], *io.get_draw_parameter_mesh(), '等間隔サンプリング: POS')
    print("外れ値除去: 統計的")
    outl_st= outlier.remove_statistical_outlier(pcd_mesh)
    io.draw_inlier_outlier_mesh(mesh, pcd_mesh, outl_st, *io.get_draw_parameter(), "外れ値除去: 統計的")
    print("外れ値除去: 半径")
    outl_rad= outlier.remove_statistical_outlier(pcd_mesh)
    io.draw_inlier_outlier_mesh(mesh, pcd_mesh, outl_rad, *io.get_draw_parameter(), "外れ値除去: 半径")
    print("法線推定")
    pcd_normal = normal.estimate_vertex_normals(pcd_mesh)
    io.draw_simple([pcd_normal], '法線推定: 点群処置後', point_show_normal=True)
    print("法線推定(mesh)")
    mesh_normal = normal.calc_vertex_normals_mesh(mesh)
    io.draw_simple([mesh_normal], '法線推定: 処置後(wireframeなし)')
    io.draw_simple_mesh([mesh_normal], '法線推定: 処置後(wireframeあり)')


num = input("1.pcd(bathtub) 2.pcd(fragment) 3.mesh(axes) 4.mesh(knot) : ")
if num == "1":
    # pcd1. bathtub -------------------------
    print("# pcd1. bathtub ##########")
    pcd_bathtub = io.load_point_cloud(file1_bathtub)
    pcd_calculation(pcd_bathtub)
elif num == "2":
    # pcd2. fragment -------------------------
    print("# pcd2. fragment ##########")
    pcd_fragment = io.load_point_cloud(file_fragment)
    pcd_calculation(pcd_fragment)
elif num == "3":
    # mesh1. 軸 -----------------------------
    print("# mesh1. axes ##########")
    mesh_axis = io.create_axes()
    mesh_calculation(mesh_axis)
elif num == "4":
    # mesh2. knot -----------------------------
    print("# mesh2. knot ##########")
    mesh_knot = io.load_mesh(file_knot)
    mesh_calculation(mesh_knot)
