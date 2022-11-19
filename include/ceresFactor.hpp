#ifndef _CERES_FACTOR_HPP_
#define _CERES_FACTOR_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

// 用于自动求导的残差块
struct PointPointFactor
{
    Eigen::Vector3d src_p, tgt_p;
    Eigen::Matrix3d M_;

    PointPointFactor(const Eigen::Vector3d& sp, 
                    const Eigen::Vector3d& tp, 
                    const Eigen::Matrix3d& M)
    : src_p(sp), tgt_p(tp), M_(M) {}

    template <typename T>
	bool operator()(const T *para, T *residual) const
    {
        Eigen::Quaternion<T> q_v{T(para[3]), T(para[0]), T(para[1]), T(para[2])};
        Eigen::Matrix<T, 3, 1> t_v{T(para[4]), T(para[5]), T(para[6])};

        Eigen::Matrix<T, 3, 1> tmp_a{T(src_p[0]), T(src_p[1]), T(src_p[2])};

        Eigen::Matrix<T, 3, 1> Ta_ = q_v*tmp_a + t_v;

        Eigen::Matrix<T, 3, 1> d_i = tgt_p - Ta_;

        // f(x) = (C^B+R*C^A*R^T)*(p^B-R*p^A-t)
        // F(x) = ||f(x)||^2
        Eigen::Matrix<T, 3, 1> Md = M_.cast<T>()*d_i;

        residual[0] = Md[0];
        residual[1] = Md[1];
        residual[2] = Md[2];

        return true;
    }
};

// 不使用Ceres定义好的参数块，我们自己定义parameter={qx, qy, qz, qw, tx, ty, tz}
class PoseLocalParameterization : public ceres::LocalParameterization 
{ 
    // 需要自己实现状态量(global state)与更新量(local state)的更新方式, x_ptr:7, delta:6, x_plus_delta_ptr:7
    virtual bool Plus(const double* x_ptr, const double* delta, double* x_plus_delta_ptr) const;
    // 需要自己实现状态量w.r.t更新量的雅克比
    virtual bool ComputeJacobian(const double *x, double *jacobian) const; 
    // 告诉ceres我们状态量和更新量的维度
    virtual int GlobalSize() const { return 7; }; 
    virtual int LocalSize() const { return 6; }; 
}; 

bool PoseLocalParameterization::Plus(const double* x_ptr,
                                           const double* delta,
                                           double* x_plus_delta_ptr) const 
{
    // 实现加法：quaternion(x_plus_delta) = quaternion(rotation_vector(delta)) * quaternion(x_ptr)
    Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);
    Eigen::Map<const Eigen::Quaterniond> x(x_ptr);

    const double norm_delta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0) {
        const double sin_delta_by_delta = sin(norm_delta) / norm_delta;

        // Note, in the constructor w is first.
        Eigen::Quaterniond delta_q(cos(norm_delta),
                                sin_delta_by_delta * delta[0],
                                sin_delta_by_delta * delta[1],
                                sin_delta_by_delta * delta[2]);
        // 注意乘法的顺序
        x_plus_delta = delta_q * x;
    } else {
        x_plus_delta = x;
    }

    // 实现加法,由于我们使用左乘，所以直接叠加
    Eigen::Map<const Eigen::Vector3d> _p(x_ptr+4);
    Eigen::Map<const Eigen::Vector3d> dp(delta+3);
    Eigen::Map<Eigen::Vector3d> p(x_plus_delta_ptr+4); 
    p = _p + dp;

    return true;
}

// 定义如何计算雅克比矩阵 
bool PoseLocalParameterization::ComputeJacobian(const double *x,  
    double *jacobian) const 
{ 
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian); 
    j.setZero();
    /*
    如果按照严格定义，也就是Ceres的源码EigenQuaternionParameterization的话，就是如下方式
    参考https://www.cnblogs.com/vivian187/p/16502590.html#3eigenquaternionparameterization
    但是实际上如果这样定义的话(四元数关于旋转向量的雅克比，维度是4*3)，根据链式法则，我们的问题的雅克比为
    df(g_x)   d g_x
    ------- * -----
    d g_x     d l_x 
    那么我们残差块处的雅克比就是残差关于四元数的雅克比，那么问题将变得复杂，所以我们提前将扰动模型得到的雅克比
    放到残差块，而参数块这边的雅克比用单位矩阵代替，数学上是合理的，然后在代码层次上也是work的
    Eigen::Matrix<double, 4, 3> dr;
    dr<<x[3], x[2], -x[1],
        -x[2], x[3], x[0],
        x[1], -x[0], x[3],
        -x[0], -x[1], -x[2];
    j.block<4, 3>(0, 0) = dr;
    j.block<3, 3>(4, 3).setIdentity();
    */

    // 这里用单位矩阵占位，注意qw那行为0，使得我们的参数维度合规
    j.block<3, 3>(0, 0).setIdentity();
    j.block<3, 3>(4, 3).setIdentity();

    return true; 
} 

// 反对称
inline Eigen::Matrix<double,3,3> skew(const Eigen::Matrix<double,3,1>& mat_in){
    Eigen::Matrix<double,3,3> skew_mat;
    skew_mat.setZero();
    skew_mat(0,1) = -mat_in(2);
    skew_mat(0,2) =  mat_in(1);
    skew_mat(1,2) = -mat_in(0);
    skew_mat(1,0) =  mat_in(2);
    skew_mat(2,0) = -mat_in(1);
    skew_mat(2,1) =  mat_in(0);
    return skew_mat;
}

// 显式求导的残差块
struct explictPointPointFactor : public ceres::SizedCostFunction<3, 7> // 残差块维度3， 第一个参数块维度7， 如果有第二、第三参数块的话就是<3, 7, 4, 3, ...>这样
{
    Eigen::Vector3d src_p, tgt_p;
    Eigen::Matrix3d M_;

    explictPointPointFactor(const Eigen::Vector3d& sp, 
                    const Eigen::Vector3d& tp, 
                    const Eigen::Matrix3d& M)
    : src_p(sp), tgt_p(tp), M_(M) {}

    // 解析求导需要实现Evaluate函数，注意para是指针的指针，因为我们有可能存在若干个参数块所有是para[i][j] 第i个参数块的第j个元素
    virtual bool Evaluate(double const * const * para, 
                            double *residuals, double **jacobians) const
    {
        Eigen::Quaternion<double> q_v{para[0][3], para[0][0], para[0][1], para[0][2]};
        Eigen::Matrix<double, 3, 1> t_v{para[0][4], para[0][5], para[0][6]};

        Eigen::Matrix<double, 3, 1> tmp_a{src_p[0], src_p[1], src_p[2]};

        Eigen::Matrix<double, 3, 1> Ta_ = q_v*tmp_a + t_v;

        Eigen::Matrix<double, 3, 1> d_i = tgt_p - Ta_;

        Eigen::Matrix<double, 3, 1> Md = M_*d_i;

        residuals[0] = Md[0];
        residuals[1] = Md[1];
        residuals[2] = Md[2];

        // 左扰动求导，要区分好我们用so(3)还是se(3)进行求导，区别很大，可以看F-LOAM[se(3)]和VINS[so(3)]的更新方式比较
        Eigen::Matrix<double, 3, 3> dfdq = M_ * skew(q_v*tmp_a);
        // https://blog.csdn.net/weixin_39910711/article/details/99445129
        Eigen::Matrix<double, 3, 3> dfdt = -M_; // 分子布局，不转置

        if(jacobians!=nullptr)
        {
            if(jacobians[0]!=nullptr)
            {
                // 残差关于第一个参数块的雅克比
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                // https://github.com/wh200720041/floam/issues/50
                // 这里要联系参数块中的全局参数关于局部参数的雅克比
                J.block<3, 3>(0, 0) = dfdq;
                // 注意要跳过q.w那一列
                J.block<3, 3>(0, 4) = dfdt;
            }

        }

        return true;
    }
    
};

#endif