import os
import open3d as o3d
import numpy as np
from lib import basis, icp_detail_sample, io, registration, point_sampling

# 事前のdownloadコード ################################# 
# dirname = "rgbd-dataset"
# classes = ["apple", "banana", "camera"]
# url="https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/"
# for i in range(len(classes)):
#     if not os.path.exists(dirname + "/" + classes[i]):
#         os.system("wget " + url + classes[i] + "_1.tar")
#         os.system("tar xvf " + classes[i] + "_1.tar")

file_bunny000 = './data/bun000.pcd'
file_bunny045 = './data/bun045.pcd'

# sin関数の近傍点探索 ##########
def dist_main():
    print('# sin関数 ##########')
    # sin関数の点群データXを生成
    X_x = np.arange(-np.pi,np.pi, 0.1)
    X_y = np.sin(X_x)
    X_z = np.zeros(X_x.shape)
    X = np.vstack([X_x, X_y, X_z]).T
    # Xをopen3Dの点群として定義
    pcd_X = o3d.geometry.PointCloud()
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_X.paint_uniform_color([0.5,0.5,0.5])

    # 点pを定義
    p = np.array([1.0,0.0,0.0])
    # pをOpen3dの点群として定義
    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector([p])
    pcd_p.paint_uniform_color([0.0,0.0,1.0])

    # 座標を用意
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # 描画
    io.draw_simple([mesh, pcd_X, pcd_p], 'sin関数')

    # 最近傍点の探索
    def dist( p, X ):
        dists = np.linalg.norm(p-X,axis=1) 
        min_dist = min(dists)
        min_idx = np.argmin(dists)
        
        return min_dist, min_idx

    min_dist, min_idx = dist(p,X)

    # 最近傍点を含め可視化
    np.asarray(pcd_X.colors)[min_idx] = [0.0,1.0,0.0]
    print("distance:{}, idx:{}".format(min_dist, min_idx))
    io.draw_simple([mesh, pcd_X, pcd_p], 'sin関数:最近傍点をマーキング')

# kd-treeによる探索 ###################
def k_tree_main():
    print('# kd-treeによる探索 ###################')
    # kd-tree構築
    pcd = io.load_point_cloud(file_bunny000)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = registration.generate_kd_tree(pcd)
    # knn
    query = 10000
    pcd.colors[query] = [1, 0, 0]
    idx, _ = registration.kd_tree_knn(pcd, pcd_tree, query, 200)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    io.draw_simple([pcd], 'kd-tree: knn')
    # radius
    query = 20000
    pcd.colors[query] = [1, 0, 0]
    idx, _ = registration.kd_tree_radius(pcd, pcd_tree, query, 0.01)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    io.draw_simple([pcd], 'kd-tree: radius')
    # hybrid
    query = 5000
    pcd.colors[query] = [1, 0, 0]
    idx, _ = registration.kd_tree_hybrid(pcd, pcd_tree, query, 200, 0.01)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 1]
    io.draw_simple([pcd], 'kd-tree: hybrid')



# ICP ################################
'''
手順1. ソース点群とターゲット点群の対応付
     * 点群データのロード
     * voxel化: ダウンサンプリング（適当な量にする)
     * 各点で近傍点探索
       - 全数探索、kd-tree(クエリのk個の近傍点を抽出、クエリから指定した半径以内、ハイブリッドなどの手法がある)
     * 各点の近傍点から特徴点を抽出
手順2. 剛体変換の推定
     * Point-to-Point: 剛体変換を適用した点とその対応点の距離の総和を評価
     * Point-to-Plane: ソースの点とターゲットの面の距離の総和を評価
手順3. 物体の姿勢のアップデート
手順4. 収束判定(収束しない場合は1へ戻る)
'''

def icp_main():
    # ファイル読み込み
    pcd_s = io.load_point_cloud(file_bunny000)
    pcd_t = io.load_point_cloud(file_bunny045)
    io.draw_simple([pcd_s, pcd_t], 'ICP: ref')
    # ボクセル化
    pcdv_s = point_sampling.boxelize_point(pcd_s, 0.005)
    pcdv_t = point_sampling.boxelize_point(pcd_t, 0.005)
    # 色指定
    pcdv_s.paint_uniform_color([0.0, 1.0, 0.0])
    pcdv_t.paint_uniform_color([0.0, 0.0, 1.0])
    # 表示
    io.draw_simple([pcdv_s, pcdv_t], 'ICP: voxel')
    # ICP
    result = registration.icp(pcdv_s, pcdv_t, 0.05)
    # 結果確認
    trans_reg = result.transformation
    print(trans_reg)  
    # registration後の点群生成
    pcd_reg = basis.transformation(pcdv_s, trans_reg)
    pcd_reg.paint_uniform_color([1.0, 0.0, 0.0])
    # 表示
    io.draw_simple([pcd_reg, pcdv_t])

def icp_detail():
    # ファイル読み込み
    pcd_s = io.load_point_cloud(file_bunny000)
    pcd_t = io.load_point_cloud(file_bunny045)
    io.draw_simple([pcd_s, pcd_t], 'ICP: ref')
    # ボクセル化
    pcdv_s = point_sampling.boxelize_point(pcd_s, 0.005)
    pcdv_t = point_sampling.boxelize_point(pcd_t, 0.005)
    # 色指定
    pcdv_s.paint_uniform_color([0.0, 1.0, 0.0])
    pcdv_t.paint_uniform_color([0.0, 0.0, 1.0])
    # 表示
    io.draw_simple([pcdv_s, pcdv_t], 'ICP: voxel')
    # ICP詳細処理
    icp_detail_sample.icp_detail(pcdv_s, pcdv_t)


num = input("選択: 1.sin関数の最近傍点探索 2.kd-treeによる探索 3.ICP 4.ICP detail: ")
if num == '1':
    dist_main()
elif num == '2':
    k_tree_main()
elif num == '3':
    icp_main()
elif num == '4':
    icp_detail()
