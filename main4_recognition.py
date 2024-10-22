import os
import numpy as np
import open3d as o3d
import copy

from lib import io, keypoint_features, basis, normal

'''
# 物体認識
* 入力データに対して、ラベルを出力する
* ラベルはあらかじめ用意しておく(人が決めるか、たくさんのデータを用意しておきそれ自体がラベルとなる)
* 共通の手順は以下
1.ラベル付与された3次元データを用意する
2.3次元データから特徴量を抽出する
3.特徴量からラベルを推定する識別器を用意(学習)する
4.認識対象物体の3次元データから特徴量を抽出する
5.識別器を用いてラベルを推定する
'''
'''
物体認識:FPFHで教示データ、テストデータの両方の特徴量を計算する
object_recognition.extract_features()
物体認識:FPFHで計算した特徴量を比較して、類似度を出す
object_recognition.onenn_classification()
'''
def extract_fpfh(filename):
    '''
    fpfhを算出し、特徴量を総和して特徴量のノルムを1に正規化して返す
    '''
    pcd = io.load_point_cloud(filename)
    pcd = pcd.voxel_down_sample(0.01)
    fpfh = keypoint_features.fpfh_features(pcd)
    sum_fpfh = np.sum(np.array(fpfh.data),1)
    return( sum_fpfh / np.linalg.norm(sum_fpfh) )

def recognition1():
    # 事前のdownloadコード ################################# 
    # 1.ラベル付与された3次元データを用意する
    dirname = "rgbd-dataset"
    classes = ["apple", "banana", "camera"]
    # url="https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/"
    # for i in range(len(classes)):
    #     if not os.path.exists(dirname + "/" + classes[i]):
    #         os.system("wget " + url + classes[i] + "_1.tar")
    #         os.system("tar xvf " + classes[i] + "_1.tar")

    # 2.3次元データから特徴量を抽出する
    # 3.特徴量からラベルを推定する識別器を用意(学習)する
    # 4.認識対象物体の3次元データから特徴量を抽出する
    nsamp = 100
    feat_train = np.zeros( (len(classes), nsamp, 33) )
    feat_test = np.zeros( (len(classes), nsamp, 33) )
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
    # 5.識別器を用いてラベルを推定する
    # k=1のk最近傍法 2つの点群データの類似度としてベクトルの内積を計算する
    # 1-NN classification
    for i in range(len(classes)):
        max_sim = np.zeros((3, nsamp))
        for j in range(len(classes)):
            sim = np.dot(feat_test[i], feat_train[j].transpose())
            # j番目の物体の全学習データの中で最も近いデータとの類似度を格納
            max_sim[j] = np.max(sim,1)  
        # 各テストデータの推定ラベルが、そのテストデータ本来のラベルiに一致する場合の個数を計算する
        correct_num = (np.argmax(max_sim,0) == i).sum()
        print ("Accuracy of", classes[i], ":", correct_num*100/nsamp, "%")


def keypoint_and_feature_extraction( pcd, voxel_size ):
    # ボクセル化
    keypoints = pcd.voxel_down_sample(voxel_size)
    radius_normal = 2.0*voxel_size
    keypoints = normal.estimate_vertex_normals(pcd=keypoints, radius=radius_normal, max_nn=30)

	# keypoints.estimate_normals(
	# 	o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 法線ベクトルの向きを原点にする
    viewpoint = np.array([0.,0.,0.], dtype='float64')
    keypoints.orient_normals_towards_camera_location( viewpoint )

    radius_feature = 5.0*voxel_size
    fpfh = keypoint_features.fpfh_features(keypoints, radius=radius_feature)
    return keypoints, fpfh

def create_lineset_from_correspondences( corrs_set, pcd1, pcd2, 
                                         transformation=np.identity(4) ):
    """ 対応点セットからo3d.geometry.LineSetを作成する．
    Args:
        result(o3d.utility.Vector2iVector) ): 対応点のidセット
        pcd1(o3d.geometry.PointCloud): resultを計算した時の点群1
        pcd2(o3d.geometry.PointCloud): resultを計算した時の点群2
        transformation(numpy.ndarray): 姿勢変換行列(4x4)
    Return:
        o3d.geometry.LineSet
    """
    pcd1_temp = copy.deepcopy(pcd1)
    pcd1_temp.transform(transformation) 
    corrs = np.asarray(corrs_set)
    np_points1 = np.array(pcd1_temp.points)
    np_points2 = np.array(pcd2.points)
    points = list()
    lines = list()

    for i in range(corrs.shape[0]):
        points.append( np_points1[corrs[i,0]] )
        points.append( np_points2[corrs[i,1]] )
        lines.append([2*i, (2*i)+1])

    colors = [np.random.rand(3) for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def recognition2(type_correspondance: str):
    '''
    # 特定物体の姿勢推定
    * 特徴量マッチング
    - 異なる視点で撮影された点群同士を張り合わせる
    - 2つの点群を貼り合わせる変換行列を求める(4x4の同時変換行列=回転+平行移動が一般的)
    - ICP補正のように点群の初期位置が近いなどの仮定はない
    - 推定する位置姿勢の精度が求められる場合は、特徴量マッチングで得られた位置姿勢を初期値としてICPアルゴを適用することが多い
    - 全ての点に対して探索すると膨大のため、よく似た部分的な領域を見つけ出して、その情報を元に姿勢を計算する
    1. 特徴点検出: ISSや等間隔サンプリング(Voxel Grid Filter)など
    2. 特徴量記述: その特徴点らしさを表現する情報を付与する. FPFHなど
    3. 対応点探索: 量特徴量のノルムの小さいペアを対応する、双方向チェックする、RatioTest(最近傍のノルムが他と比べて際立って小さいか調べる)
    4. 姿勢計算
    '''
    zoom=0.3199
    front = [0.024, -0.225, -0.973]
    lookat = [0.488, 1.722, 1.556]
    up = [0.047, -0.972, 0.226]
    path = r'3rdparty/Open3D/examples/test_data/ICP/'
    pcd_src = io.load_point_cloud(path+'cloud_bin_0.pcd')
    pcd_tgt = io.load_point_cloud(path+'cloud_bin_1.pcd')
    pcd_src.paint_uniform_color([0.5,0.5,1.0])
    pcd_tgt.paint_uniform_color([1.0,0.5,0.5])
    # 比較するためsrc位置をずらす
    initial_trans = np.identity(4)
    initial_trans[0,3] = -3.0
    pcd_src_trns = basis.transformation(pcd_src, initial_trans)
    # 描画
    io.draw_detail([pcd_src_trns, pcd_tgt], zoom=zoom, front=front, lookat=lookat, up=up, window_name='recognition2-1')

    # 1. 特徴点検出: ISSや等間隔サンプリング(Voxel Grid Filter)など, 今回はvoxel
    # 2. 特徴量記述: その特徴点らしさを表現する情報を付与する. FPFHなど, 今回はFPFH
    voxel_size = 0.1
    s_kp, s_feature = keypoint_and_feature_extraction( pcd_src, voxel_size )
    t_kp, t_feature = keypoint_and_feature_extraction( pcd_tgt, voxel_size )
    # 描画
    s_kp.paint_uniform_color([0,1,0])
    t_kp.paint_uniform_color([0,1,0])
    s_kp_trns = basis.transformation(s_kp, initial_trans)
    io.draw_detail([pcd_src_trns,s_kp_trns, pcd_tgt,t_kp], zoom=zoom, front=front, lookat=lookat, up=up, window_name='recognition2-2')

    # 3. 対応点探索: RatioTest
    # 特徴量ベクトルの取り出し(n,33)行列を作成
    np_s_feature = s_feature.data.T
    np_t_feature = t_feature.data.T
    # 対応点のセットを保存するベクトル定義
    corrs = o3d.utility.Vector2iVector()
    threshold = 0.9
    for i,feat in enumerate(np_s_feature):
        # source側の特定の特徴量とtarget側の全特徴量間のL2ノルムを計算
        distance = np.linalg.norm( np_t_feature - feat, axis=1 )
        nearest_idx = np.argmin(distance)
        dist_order = np.argsort(distance)
        # 1,2位の日を計算してth以下であれば正しい対応点とみなす
        ratio = distance[dist_order[0]] / distance[dist_order[1]]
        if ratio < threshold:
            corr = np.array( [[i],[nearest_idx]], np.int32 )
            corrs.append( corr )
    print('対応点セットの数：', (len(corrs)) )

    # 4. 姿勢計算: 
    # 全点利用
    if type_correspondance=='all':
        # 対応関係の描画
        line_set = create_lineset_from_correspondences( corrs, s_kp, t_kp, initial_trans )
        io.draw_simple([pcd_src_trns, s_kp_trns, pcd_tgt,t_kp, line_set], window_name='姿勢計算: 対応関係')
        # 2つの対応が取れた点群の二乗誤差を最小化する変換行列を算出
        trans_ptp = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        # 変換行列を作成(全点なので外れ値も含む)
        trans_all = trans_ptp.compute_transformation( s_kp, t_kp, corrs )
        # 変換して重ねて表示
        pcd_corr = basis.transformation(pcd_src, trans_all)
        io.draw_simple([pcd_corr, pcd_tgt], window_name='2点群の張り合わせ: 全点利用')
    # RANSAC RANdom SAmple Consensus
    else:
        distance_threshold = voxel_size*1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                s_kp, # ソース側の特徴点
                t_kp, # ターゲット側の特徴てん
                corrs, # 対応て探索によって得られた対応点のインデックス
                distance_threshold, # インライアと判定する距離(マージン)の閾値
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n = 3, # 姿勢変換行列の計算のためにサンプリングする対応点の個数
                checkers = [ # 枝刈処理(=サンプリングと評価の間)に使われる条件
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), # 任意の対応2点のsrc, tgt点群内距離が近いと有望変換行列とみなす
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold) # 距離が近いものを有望変換行列とみなす
                ], 
                criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # RANSACの終了条件 arg1=試行最大数, arg2=早期終了の条件閾値 
                )
        '''
        resultの内容
        correspondense_set: インライアと判定された対応点のインデックスのリスト
        fitness           : インライア数/対応点数 大きいほどよい
        inlier_rmse       : インライアの平均二乗誤差 小さいほどよい
        transformation    : 4x4の変換行列 
        '''
        line_set = create_lineset_from_correspondences( result.correspondence_set, s_kp, t_kp, initial_trans )
        io.draw_simple([pcd_src_trns, s_kp_trns, pcd_tgt,t_kp, line_set], window_name='姿勢計算: RANSACで抽出したインライア対応関係')
        pcd_corr = basis.transformation(pcd_src, result.transformation)
        io.draw_simple([pcd_corr, pcd_tgt], window_name='2点群の張り合わせ: RANSAC')

'''
# 一般物体認識
* Bag Of Features(BoF)の処理の流れ
  1.DB内の物体データから点をサンプリングする
  2.各点における局所特徴量を抽出する
  3.全ての局所特徴量に対してk-meansクラスタリングを行う
  4.各クラスタ中心(visual word)の集合をvisual codebookをする
  5.入力データから点をサンプリングする
  6.各点における局所特徴量を抽出し、最近傍のvisual wordに割り当てる
  7.各visual wordに割り当てられた点の個数を数え上げ、ヒストグラムを作る
  - SHIFT特徴量がよく使われる
  - このヒストグラムの類似度を算出する（KLダイバージェンス）
* Object-Pose Tree(OP-Tree)
  - RGB-D Object Datasetが使われている
  - 各画像にカテゴリ、インスタンス、視点、姿勢のラベルを付与する
  - 特徴量としてSHIFT(BoFとは少し異なる)統計量を計算
  - 特徴量に対して各ラベルを線形識別器で識別する  
  '''

num = input("選択: 1.認識試行1, 2.物体認識(姿勢計算=all), 3.物体認識(姿勢計算=ransac) :")
if num == '1':
    recognition1()
elif num == '2':
    recognition2('all')
elif num == '3':
    recognition2('ransac')
