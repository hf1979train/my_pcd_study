import numpy as np
import open3d as o3d


# ファイルIO  #####################################
# 点群ファイルの読み込み
def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    print(pcd)
    print(np.asarray(pcd.points))
    return pcd

# メッシュファイルの読み込み
def load_mesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    print(mesh)
    return mesh

# 軸作成 ##########################################
# 軸作成
def create_axes():
    return o3d.geometry.TriangleMesh.create_coordinate_frame()


# 描画  ###########################################
# シンプルな表示
def draw_simple(pcds: list, window_name: str='Open3D', point_show_normal=False):
    o3d.visualization.draw_geometries(pcds, window_name=window_name, point_show_normal=point_show_normal)

# シンプルな表示 メッシュ用
def draw_simple_mesh(meshs: list, window_name: str='Open3D', point_show_normal=False):
    o3d.visualization.draw_geometries(meshs, window_name=window_name, mesh_show_wireframe=True, point_show_normal=point_show_normal)

# 複雑な表示
def draw_detail(pcds: list, zoom: float, front: list, lookat: list, up: list, window_name: str='Open3D', point_show_normal=False):
    o3d.visualization.draw_geometries(pcds, zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up, window_name=window_name, point_show_normal=point_show_normal)

# 複雑な表示 メッシュ用
def draw_detail_mesh(pcds: list, zoom: float, front: list, lookat: list, up: list, window_name: str='Open3D', point_show_normal=False):
    o3d.visualization.draw_geometries(pcds, zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up, window_name=window_name, mesh_show_wireframe=True, point_show_normal=point_show_normal)
    
# 調整した視点パラメータの取得
ZOOM_ADJUSTED   = 0.3412
FRONT_ADJUSTED  = [0.4257, -0.2125, -0.8795]
LOOKAT_ADJUSTED = [2.6172, 2.0475, 1.532]
UP_ADJUSTED     = [-0.0694, -0.9768, 0.2024]
def get_draw_parameter():
    '''
    zoom, front, lookat, upを返す
    '''
    return ZOOM_ADJUSTED, FRONT_ADJUSTED, LOOKAT_ADJUSTED, UP_ADJUSTED

def get_draw_parameter_mesh():
    '''
    zoom, front, lookat, upを返す
    '''
    return 1, FRONT_ADJUSTED, LOOKAT_ADJUSTED, UP_ADJUSTED

# 
def draw_inlier_outlier(cloud, ind, zoom: float, front: list, lookat: list, up: list, window_name: str='Open3D'):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up, window_name=window_name)

def draw_inlier_outlier_mesh(mesh, cloud, ind, zoom: float, front: list, lookat: list, up: list, window_name: str='Open3D'):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([5, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([mesh, outlier_cloud],
                                      zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up, window_name=window_name, mesh_show_wireframe=True)

# 変換 ###########################################
def convert_mesh2pcd(mesh):
    pcd = o3d.geometry.PointCloud()
    # メッシュの頂点を点群データとする
    pcd.points = mesh.vertices
    return pcd

