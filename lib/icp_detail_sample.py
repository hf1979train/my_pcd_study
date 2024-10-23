import sys
import open3d as o3d


from lib.icp_registration import ICPRegistration_PointToPoint
from lib.icp_registration import ICPRegistration_PointToPlane
from lib.icp_registration import visualize_icp_progress

CI_MODE_P2POINT = 0
CI_MODE_P2PLANE = 1

mode = CI_MODE_P2POINT


def icp_detail(pcd_s, pcd_t):
    # ICPの実行．
    reg = ICPRegistration_PointToPoint(pcd_s, pcd_t)
    if mode == 1:
        print("Run Point-to-plane ICP algorithm")
        reg = ICPRegistration_PointToPlane(pcd_s, pcd_t)
    # パラメータ設定
    reg.set_th_distance( 0.003 )
    reg.set_n_iterations( 100 )
    reg.set_th_ratio( 0.999 )
    # レジストレーション(ICP)
    pcd_reg = reg.registration()

    print("# of iterations: ", len(reg.d))
    print("Registration error [m/pts.]:", reg.d[-1] )
    print("Final transformation \n", reg.final_trans )

    # ICP前後を描画
    pcd_reg.paint_uniform_color([1.0,0.0,0.0])
    o3d.visualization.draw_geometries([pcd_t, pcd_reg] )

    # ICPの実行の様子の可視化
    visualize_icp_progress( reg )
