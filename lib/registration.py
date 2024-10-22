import os
import copy
import open3d as o3d
import numpy as np
from lib import keypoint_features


# kd-treeの構築 ##########
def generate_kd_tree(pcd):
    return o3d.geometry.KDTreeFlann(pcd)

# kd-tree探索: knn
def kd_tree_knn(pcd, pcd_tree, query, knn):
    '''
    kd_treeを用いて指定の点=queryの近傍点を近い順に抽出する
    args
      query: 選択した点
      knn: 抽出する点数
    return
      idx: 抽出した番号
      d: 距離の二乗
    '''
    [k, idx, d] = pcd_tree.search_knn_vector_3d(pcd.points[query], knn)
    return idx, d

# kd-tree探索: radius
def kd_tree_radius(pcd, pcd_tree, query, radius):
    '''
    kd_treeを用いて指定の点=queryから距離radius以内の点を抽出する
    args
      query: 選択した点
      knn: 抽出する点のqueryとの距離閾値
    return
      idx: 抽出した番号
      d: 距離の二乗
    '''
    [k, idx, d] = pcd_tree.search_radius_vector_3d(pcd.points[query], radius)
    return idx, d

# kd-tree探索: hybrid
def kd_tree_hybrid(pcd, pcd_tree, query, knn, radius):
    '''
    kd_treeを用いて指定の点=queryから距離radius以内の点を抽出する
    args
      query: 選択した点
      knn: 抽出する点のqueryとの距離閾値
    return
      idx: 抽出した番号
      d: 距離の二乗
    '''
    [k, idx, d] = pcd_tree.search_hybrid_vector_3d(pcd.points[query], max_nn=knn, radius=radius)
    return idx, d


# ICP ##########
'''
手順1. ソース点群とターゲット点群の対応付
手順2. 剛体変換の推定
手順3. 物体の姿勢のアップデート
手順4. 収束判定(収束しない場合は1へ戻る)
'''

def icp(pcd_source, pcd_target, threshold):
    '''
    args:
      threshold: 2つの点群を対応付する時の最大距離
    return:
      位置合わせの結果はresult.transformationに格納されている
      4x4の同次変換行列、3x3の開店行列と平行移動ベクトルで構成
    ICPの引数
      trans_init: 初期位置
      obj_func: 目的関数 point-to-point:対応付された転換の距離(二乗和):o3d.pipelines.registration.TransformationEstimationPointToPoint()
          point-to-plane:ソースの点とターゲットの面の距離を評価(面=法線):o3d.pipelines.registration.TransformationEstimationPointToPlane()

      
    '''
    trans_init = np.identity(4)
    obj_func = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = o3d.pipelines.registration.registration_icp( pcd_source, pcd_target,
                                                          threshold,
                                                          trans_init,
                                                          obj_func
                                                    )
    return result

'''
* 特定物体認識と一般物体認識
  - 特定物体認識: たくさんの3Dモデルデータから、入力データの物体と一致する(近い)モデルを特定する
  - 一般物体認識: ラベルを必要に応じて人間があらかじめ決めておき、入力データに対してラベルを出力する

* 特定物体認識と一般物体認識の手順
1. ラベルが付与された3次元データを用意する
2. 3次元データから特徴量を抽出する
3. 特徴量からラベルを推定する識別器を用意（学習）する
4. 認識対象物体の3次元データから特徴量を抽出する
5. 識別器を用いてラベルを推定する
'''

dirname = "rgbd-dataset"
classes = ["apple", "banana", "camera"]
nsamp = 100
feat_train = np.zeros( (len(classes), nsamp, 33) )
feat_test = np.zeros( (len(classes), nsamp, 33) )



def extract_fpfh( filename ):
    '''
    ファイルから点群データを読み込む
    ダウンサンプリングと法線ベクトル推定
    FPFH特徴量の抽出
    -通常は33次元の局所特徴量だが、点群でーた全体を1つの物体として認識するために
     点群データ全体から帯域特徴量を抽出する必要がある
    -全ての点から抽出されるFPFH特徴量の総和を計算して、特徴量のノルムを1に正規化して返す
    '''
    print (" ", filename)
    pcd = o3d.io.read_point_cloud(filename)
    pcd = pcd.voxel_down_sample(0.01)
    pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100))
    sum_fpfh = np.sum(np.array(fpfh.data),1)
    return( sum_fpfh / np.linalg.norm(sum_fpfh) )

# 特徴量出力：Extract features
def extract_features():
    '''
    x=1を学習データ、x=4をテストデータとして特徴量を格納していく
    '''
    for i in range(len(classes)):
        print ("Extracting train features in " + classes[i] + "...")
        for n in range(nsamp):
            filename = dirname + "/" + classes[i] + "/" + classes[i] + \
                    "_1/" + classes[i] + "_1_1_" + str(n+1) + ".pcd"
            feat_train[ i, n ] = extract_fpfh( filename )
        print ("Extracting test features in " + classes[i] + "...")
        for n in range(nsamp):
            filename = dirname + "/" + classes[i] + "/" + classes[i] + \
                    "_1/" + classes[i] + "_1_4_" + str(n+1) + ".pcd"
            feat_test[ i, n ] = extract_fpfh( filename )

# 1-NN classification
def onenn_classification():
    '''
    k=1のk最近傍法
    学習データ、テストデータと鬼33次元の特徴ベクトルで表されている
    2つの点群データの類似度としてベクトルの内積を用いる
    '''
    for i in range(len(classes)):
        max_sim = np.zeros((3, nsamp))
        for j in range(len(classes)):
        # 2つの点群データの類似度として、ベクトルの内積を用いる
            sim = np.dot(feat_test[i], feat_train[j].transpose())
            # max_simは全てのテストデータに対して、j番目の物体の全学習データの中で
            # 最も近いデータとの類似度を格納している
            # この類似度が最も高いクラスが推定されたラベルとなる
            max_sim[j] = np.max(sim,1)
        correct_num = (np.argmax(max_sim,0) == i).sum()
        print ("Accuracy of", classes[i], ":", correct_num*100/nsamp, "%")

# 特徴点抽出＆特徴量出力
def keypoint_and_feature_extraction( pcd, voxel_size ):
    '''
    pcd: 処理したい点群データ
    voxel_size: ボクセルサイズ（＝検出する特徴点の感覚）
    '''
    print("# 特徴点抽出＆特徴量出力 #####")
    # 描画1: ベース点群 ---------------------------------------
    print("描画: ベース点群")
    o3d.visualization.draw_geometries([pcd])   

    # 特徴点抽出:voxel grid filter ---------------------------
    print("# 特徴点抽出: voxel grid filter")
    keypoints = pcd.voxel_down_sample(voxel_size)
    # 描画2: voxel data
    o3d.visualization.draw_geometries([keypoints])   

    # 法線推定 ----------------------------------------------
    # 間引いたデータの倍の半径を指定する
    print("# 法線推定")
    radius_normal = 2.0*voxel_size  
    # 半径の範囲内で最大30点に対して計算する
    keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([keypoint_features.keypoints_to_spheres(keypoints), pcd])

    # 法線計算 ----------------------------------------------
    # 今回は適当な視点=原点を仮定し、その方向に法線が向くように法線方向に反転処理を施す
    print("# 法線計算")
    viewpoint = np.array([0.,0.,0.], dtype='float64')
    keypoints.orient_normals_towards_camera_location( viewpoint )
     # 描画3: 法線データ
    o3d.visualization.draw_geometries([keypoints])     

    # 特徴量の計算 ------------------------------------------
    radius_feature = 5.0*voxel_size
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return keypoints, feature
