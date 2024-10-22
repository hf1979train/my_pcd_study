import sys
import numpy as np
import open3d as o3d
import copy

'''
* 回転
* 並進
* スケーリング
'''
# 
# 回転
def rotate(mesh_basis, rx, ry, rz):
    '''
    pcd, meshどちらもOK
    '''
    R = o3d.geometry.get_rotation_matrix_from_yxz([rx, ry, rz])
    print ("R:",np.round(R,7))
    mesh_rotate = copy.deepcopy(mesh_basis)
    mesh_rotate.rotate(R, center=[0,0,0])
    return mesh_rotate

# 並進
def translate(mesh_basis, tx, ty, tz):
    '''
    pcd, meshどちらもOK
    '''
    t = [tx, ty, tz]
    mesh_translate = copy.deepcopy(mesh_basis).translate(t)
    return mesh_translate

# スケーリング
def scaling(mesh_basis, scale):
    '''
    pcd, meshどちらもOK
    '''
    mesh_scale = copy.deepcopy(mesh_basis)
    mesh_scale.scale(scale, center=mesh_scale.get_center())
    return mesh_scale

# 座標変換
def transformation(pcd, transformation=np.identity(4)):
    '''
    transformation; 4x4行列,初期値は変換しない値
    '''
    pcd_t = copy.deepcopy(pcd)
    pcd_t.transform(transformation)
    return pcd_t