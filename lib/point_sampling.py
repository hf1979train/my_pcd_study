import numpy as np
import open3d as o3d

# 等間隔サンプリング：ボクセル化
def boxelize_point(pcd, voxel_size: float):
    '''
    voxel_size: ボクセルデータの1辺の大きさ
    '''
    print(pcd)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(downpcd)
    return downpcd


# 等間隔サンプリング: Farthest Point Sampling（FPS）
def _l2_norm(a, b):
    return ((a - b) ** 2).sum(axis=1)

def fps_point(pcd, k: int, metrics=_l2_norm):
    '''
    FPSの流れ
    * はじめに１点をランダムに選択します
    * 次に，この点と他のすべての点との距離を計算し，最も距離の大きい点を選択します
    * そして，今回選択された点と他のすべての点との距離を計算します
    * 各点について，最初に選ばれた点との距離と，今回選ばれた点との距離のうち，より近いほうの（最小となる）距離に注目し，この値が最大となる点を第三の点として選択します
    * この操作を繰り返して k 個の点を選択します．
    args
      k: サンプリング点数
    '''
    print(pcd) 
    # FPS
    indices = np.zeros(k, dtype=np.int32)
    points = np.asarray(pcd.points)
    distances = np.zeros((k, points.shape[0]), dtype=np.float32)
    indices[0] = np.random.randint(len(points))
    farthest_point = points[indices[0]]
    min_distances = metrics(farthest_point, points)
    distances[0, :] = min_distances
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = points[indices[i]]
        distances[i, :] = metrics(farthest_point, points)
        min_distances = np.minimum(min_distances, distances[i, :])
    downpcd = pcd.select_by_index(indices)
    print(downpcd)
    return downpcd

# 等間隔サンプリング: PoissonDisk Sampling（POS）
def pos_point_mesh(mesh, k: int):
    '''
    POS: ランダムサンプリング、サンプリングされた2点間の距離の最小値を指定した値以下にならないようにサンプリングする
    制約: 入力ファイルはmeshデータであること(open3dの関数の制約)
    args: k=サンプリング数
    '''
    print(mesh)
    downpcd = mesh.sample_points_poisson_disk(number_of_points=k)
    print(downpcd)
    return downpcd
