import sys
import open3d as o3d
import numpy as np

from lib import normal

'''


'''

# 特徴点の描画見栄え用のレンダリング処理
def keypoints_to_spheres(keypoints):
    try:
        spheres = o3d.geometry.TriangleMesh()
        pts = np.asarray(keypoints.points)
        for keypoint in pts:
        # for keypoint in keypoints.points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.translate(keypoint)
            spheres += sphere
        spheres.paint_uniform_color([1.0, 0.75, 0.0])
    except Exception as e: 
        print(keypoint)
    return spheres


# 特徴点 Harris3D
def harris3d_keypoints( pcd, radius=0.01, max_nn=10, threshold=0.001 ):
    '''
    harris指標Rを計算する: エッジ上の点はR<<0, コーナー点はR>>0. そのほかはRは小さな正の値
    各点に対してある範囲の近傍点をチェックし、自身のR値よりも小さいR値を持つ点を消去する
    (重複検出の防止=NMS, Non Maximum Suppression)
    参考:PointCloudLibraryの関数HarrisKeypoint3D()は、画素勾配の代わりに法線ベクトルを用いている
    '''
    print(pcd)
    # 法線推定
    pcd = normal.estimate_vertex_normals(pcd, radius=radius, max_nn=max_nn)
    # 近傍点探索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    harris = np.zeros( len(np.asarray(pcd.points)) )
    is_active = np.zeros( len(np.asarray(pcd.points)), dtype=bool )

    # 各点の近傍点群の法線の共分散(Harris response)を計算
    for i in range( len(np.asarray(pcd.points)) ):
        [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], max_nn)
        pcd_normals = pcd.select_by_index(inds)
        pcd_normals.points = pcd_normals.normals
        [_, covar] = pcd_normals.compute_mean_and_covariance()
        harris[ i ] = np.linalg.det( covar ) / np.trace( covar )
        if (harris[ i ] > threshold):
            is_active[ i ] = True

    # NMS(non maximum suppression)で最大値でないものを消去する
    for i in range( len(np.asarray(pcd.points)) ):
        if is_active[ i ]:
            [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], max_nn)
            inds.pop( harris[inds].argmax() )
            is_active[ inds ] = False

    keypoints = pcd.select_by_index(np.where(is_active)[0])
    print(keypoints)   
    return keypoints

# 特徴点抽出: Intrinsic Shape Signature ISS
def iss_keypoint(pcd):
    '''
    各点で近傍点の重心を算出する。
    各点の、重心を使った共分散行列Mを計算し、Mの最小固有値を顕著度として用いる。
    λ_n+1/λ_nがある閾値異常となる点は除外（主成分軸が同じような分線を持つ場合は特徴点抽出処理が不安定になる）
    NMS処理を行い、近傍点群の中で顕著度が最大となる点のみを抽出する
    '''
    print(pcd)
    # 特徴点抽出:ISS
    keypoints = \
    o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                salient_radius=0.005,
                                                non_max_radius=0.005,
                                                gamma_21=0.5,
                                                gamma_32=0.5)
    print(keypoints)
    return keypoints

# 局所特徴量 FPFH Fast Point Feature Histograms
def fpfh_features(pcd, radius=0.03, max_nn=100):
    '''
    PFH
    ある特徴点pを中心とした小球領域に含まれるk個の近傍点を求める
    それらの点から2点を選ぶ組み合わせに対してパラメータalpha, phi, thetaのヒストグラムを計算する
    パラメータは2点の法線ベクトルで計算する
    SPFH(Simple Point Feature Histgrams)
    2点間ではなく特徴点pとk近傍点間のみを計算するやり方
    FPFH
    SPFHと重み付wの関数として定義される
    ヒストグラムはそれぞれ11ビンあり、各特徴点で33次元の特徴量が出力される
    '''
    # 法線ベクトルの計算
    pcd = normal.estimate_vertex_normals(pcd)
    # FPFHの計算
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    print(fpfh)
    print(fpfh.data)
    return fpfh

# 通常は特徴点のみに特徴量を計算することが多い