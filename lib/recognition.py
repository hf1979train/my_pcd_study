import open3d as o3d
import numpy as np

# 平面のセグメント化
def segment_plane(pcd, distance_threshold=0.005, ransac_n=3, num_iterations=500):
    '''
    return
      plane_model: 平面の式 ax+by+cz+dの[a,b,c,d]
      inliers: 平面の点のリスト
    '''
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    return plane_model, inliers


def ComputeSphereCoefficient( p0, p1, p2, p3 ):
    """ 与えられた4点を通る球の方程式のパラメータ(a,b,c,r)を出力する．
        解が求まらない場合は0行列を出力
    Args:
      p0,p1,p2,p3(numpy.ndarray): 4 points (x,y,z)
    Return:
      Sphere coefficients.
    """

    A = np.array([p0-p3,p1-p3,p2-p3])
    
    p3_2 = np.dot(p3,p3)
    b = np.array([(np.dot(p0,p0)-p3_2)/2,
                  (np.dot(p1,p1)-p3_2)/2,
                  (np.dot(p2,p2)-p3_2)/2])
    coeff = np.zeros(3)
    try:
        ans = np.linalg.solve(A,b)
    except:
        print( "!!Error!! Matrix rank is", np.linalg.matrix_rank(A) )
        print( "  Return", coeff )
        pass
    else:
        tmp = p0-ans
        r = np.sqrt( np.dot(tmp,tmp) )
        coeff = np.append(ans,r)

    return coeff

def EvaluateSphereCoefficient( pcd, coeff, distance_th=0.01 ):
    """ 球の方程式の係数の当てはまりの良さを評価する．
    Args:
      pcd(numpy.ndarray): Nx3 points
      coeff(numpy.ndarray): shpere coefficient. (a,b,c,r)
      distance_th(float):
    Returns:
      fitness: score [0-1]. larger is better
      inlier_dist: smaller is better
      inliers: indices of inliers
    """
    fitness = 0 # インライア点数/全点数
    inlier_dist = 0 #インライアの平均距離
    inliers = None #インライア点の番号セット
    
    dist = np.abs( np.linalg.norm( pcd - coeff[:3], axis=1 ) - coeff[3] )
    n_inlier = np.sum(dist<distance_th)
    if n_inlier != 0:
        fitness = n_inlier / pcd.shape[0]
        inlier_dist = np.sum((dist<distance_th)*dist)/n_inlier
        inliers = np.where(dist<distance_th)[0]
    
    return fitness, inlier_dist, inliers

