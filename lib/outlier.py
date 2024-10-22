import open3d as o3d

# 外れ値除去: 統計的
def remove_statistical_outlier(pcd, nb_neighbors=20, std_ratio=2.0):
    '''
    各点と近傍点との距離の平均値を算出する
    平均値に基づいてある閾値を求め、近傍点との距離が閾値以上となる点を外れ値として定義
    args: nb_neighbors=考慮する近傍点の個数
          std_ratio=距離の閾値を決定するパラメータ
    '''
    print(pcd)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return ind

# 外れ値除去: 半径
def remove_radius_outlier(pcd, nb_points=16, radius=0.02):
    '''
    各点を中心とする半径の球内の点を近傍点として近傍点の個数が閾値未満となる点を外れ値として定義
        args: nb_points=閾値となる近傍点の個数
              radius=球の半径
    '''
    print(pcd)
    cl, ind = pcd.remove_radius_outlier(nb_points, radius)
    return ind
