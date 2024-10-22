import numpy as np
import open3d as o3d
import copy

from lib import io

# 法線推定
'''
点群データを想定
* 各点の近傍点を求め、近傍点群の3次元座標に対して主成分分析を行う
* 主成分分析は、近傍点群の分散教分散を求め、その固有値分解を行う分析手法
* 固有値が小さい(分散が小さい)軸を選び、これが法線ベクトルとなる
'''
def estimate_vertex_normals(pcd, radius=10.0, max_nn=10):
    pcd_normal = copy.deepcopy(pcd)
    pcd_normal.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # 描画：法線ベクトル
    print(np.asarray(pcd_normal.normals))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    return pcd_normal

# 法線計算
'''
メッシュデータを想定
* ある頂点の全ての面の三角メッシュ全てに対して、構成する辺のうち2本の外積を求める
* 全ての面の法線ベクトルの平均値が解
'''

def calc_vertex_normals_mesh(mesh):
    '''
    メッシュデータのみ対応
    '''
    mesh_normal = copy.deepcopy(mesh)
    print(mesh_normal)
    # メッシュデータから法線を計算する
    mesh_normal.compute_vertex_normals()
    # 描画：法線あり
    print(np.asarray(mesh_normal.triangle_normals))
    return mesh_normal