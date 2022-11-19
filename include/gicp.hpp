#ifndef _GICP_HPP_
#define _GICP_HPP_

#include <iostream>
#include <ceres/ceres.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "ceresFactor.hpp"
namespace gicp{

using namespace Eigen;
using namespace pcl;
using namespace std;
template <typename PointType>
class GeneralizedICP
{
public:

typedef std::shared_ptr<gicp::GeneralizedICP<PointType>> Ptr;

using PointCloudPtr = typename pcl::PointCloud<PointType>::Ptr;
using KdTreeFLANNPtr = typename pcl::KdTreeFLANN<PointType>::Ptr;

GeneralizedICP()
  :k_correspondences_(20)
  , gicp_epsilon_(0.001)
  , max_outer_iterations_(40)
  , min_number_correspondences_(4)
  , corr_dist_threshold_(5)
  , max_inner_iterations_(6)
  , inner_parameter_tolerance_(1e-3)
  , inner_gradient_tolerace_(1e-3)
  , optimization_epsilon_(1e-4)
{
  kdtreeSource.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeTarget.reset(new pcl::KdTreeFLANN<PointType>());

  num_threads_ = omp_get_max_threads();

  cost_last = DBL_MAX;
}

virtual ~GeneralizedICP()
{
}

void setInputSource(const PointCloudPtr cloud);
void setInputTarget(const PointCloudPtr cloud);
void computeCovariances(const PointCloudPtr cloudptr, const KdTreeFLANNPtr tree, 
        shared_ptr<vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>> covs);
bool align(const Matrix4d& initGuess, const PointCloudPtr aligned);
Matrix4d getTransformation();
double getScore();

inline void setKCorrespondences(const int& val){k_correspondences_ = val;}
inline void setGicpEpsilon(const double& val){gicp_epsilon_ = val;}
inline void setMaxOuterIterations(const int& val){max_outer_iterations_ = val;}
inline void setMinNumberCorrespondences(const int& val){min_number_correspondences_ = val;}
inline void setCorrDistThreshold(const double& val){corr_dist_threshold_ = val;}
inline void setInnerParameterTolerance(const double& val){inner_parameter_tolerance_ = val;}
inline void setInnerGradientTolerance(const double& val){inner_gradient_tolerace_ = val;}
inline void setOptimizationEpsilon(const double& val){optimization_epsilon_ = val;}

protected:
// 需要什么属性：
/** \brief The number of neighbors used for covariances computation.
   * default: 20
   */
  int k_correspondences_;

  /** \brief The epsilon constant for gicp paper; this is NOT the convergence
   * tolerance
   * default: 0.001
   */
  double gicp_epsilon_;

  /** \brief base transformation */
  Matrix4d base_transformation_;

  /** \brief Temporary pointer to the source dataset. */
  PointCloudPtr tmp_src_;

  /** \brief Temporary pointer to the target dataset. */
  PointCloudPtr tmp_tgt_;

  /** \brief Temporary pointer to the source dataset indices. */
  pcl::IndicesPtr tmp_idx_src_;

  /** \brief Temporary pointer to the target dataset indices. */
  pcl::IndicesPtr tmp_idx_tgt_;

  //kd-tree
  KdTreeFLANNPtr kdtreeSource;
  KdTreeFLANNPtr kdtreeTarget;

  /** \brief Input cloud points covariances. */
  shared_ptr<vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>> source_covariances_;

  /** \brief Target cloud points covariances. */
  shared_ptr<vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>> target_covariances_;

  /** \brief maximum number of outer optimizations */
  int max_outer_iterations_;

  /** \brief maximum number of inner optimizations */
  int max_inner_iterations_;

  /** \brief unused */
  int min_number_correspondences_;

  /** \brief correspondent distance threshold */
  double corr_dist_threshold_;

  /** \brief openMP */
  int num_threads_;

  // See Ceres-solver  Solver::Options
  double inner_parameter_tolerance_;
  double inner_gradient_tolerace_;

  // 平均匹配距离
  double cost_last;

  double optimization_epsilon_;
};

template <typename PointType>
void GeneralizedICP<PointType>::setInputSource(const PointCloudPtr cloud)
{
    // 拷贝一份Source点云
    tmp_src_ = cloud;
    // 构建kd tree
    kdtreeSource->setInputCloud(tmp_src_);
    // 计算每个点的分布协方差
    
    // 确保cloud的数量大于k_correspondences_
    if(tmp_src_->size()<k_correspondences_)
    {
      ROS_ERROR("source.size()<k_correspondences_, EXIT!");
      return;
    }

    source_covariances_.reset(new vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(tmp_src_->size()));
    // source_covariances_->resize(tmp_src_->size()); // 这样resize会越界

    computeCovariances(tmp_src_, kdtreeSource, source_covariances_);
}

template <typename PointType>
void GeneralizedICP<PointType>::setInputTarget(const PointCloudPtr cloud)
{
    tmp_tgt_ = cloud;
    kdtreeTarget->setInputCloud(tmp_tgt_);

    // 确保cloud的数量大于k_correspondences_
    if(tmp_tgt_->size()<k_correspondences_)
    {
      ROS_ERROR("target.size()<k_correspondences_, EXIT!");
      return;
    }

    target_covariances_.reset(new vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(tmp_tgt_->size()));
    // target_covariances_->resize(tmp_src_->size());

    computeCovariances(tmp_tgt_, kdtreeTarget, target_covariances_);

}

template <typename PointType>
void GeneralizedICP<PointType>::computeCovariances(const PointCloudPtr cloudptr, const KdTreeFLANNPtr tree, 
        shared_ptr<vector<Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>> covs)
{
  #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for(int i=0; i<cloudptr->size(); i++)
  {
    // 均值
    Eigen::Vector3d mean;
    std::vector<int> pointSearchInd;
    pointSearchInd.reserve(k_correspondences_);
    std::vector<float> pointSearchSqDis;
    pointSearchSqDis.reserve(k_correspondences_);

    Matrix3d &cov = (*covs)[i];
    mean.setZero();
    cov.setZero();
    tree->nearestKSearch(cloudptr->points[i], k_correspondences_, pointSearchInd, pointSearchSqDis);

    for(int j=0; j<k_correspondences_; j++)
    {
      PointType& pt = cloudptr->points[pointSearchInd[j]];
      mean[0] += pt.x;
      mean[1] += pt.y;
      mean[2] += pt.z;

      cov(0, 0) += pt.x * pt.x;

      cov(1, 0) += pt.y * pt.x;
      cov(1, 1) += pt.y * pt.y;

      cov(2, 0) += pt.z * pt.x;
      cov(2, 1) += pt.z * pt.y;
      cov(2, 2) += pt.z * pt.z;
    }

    // 计算实际的均值和协方差
    mean /= static_cast<double>(k_correspondences_);
    // Get the actual covariance
    for (int k = 0; k < 3; k++)
    {
      for (int l = 0; l <= k; l++) {
        cov(k, l) /= static_cast<double>(k_correspondences_);
        cov(k, l) -= mean[k] * mean[l];
        cov(l, k) = cov(k, l);
      }
    }

    // Compute the SVD (covariance matrix is symmetric so U = V')
    // 对应论文注释1
    // 协方差矩阵是对称的，所以SVD分解后，其U=V’
    // 对协方差矩阵进行SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
    cov.setZero();
    Eigen::Matrix3d U = svd.matrixU();
    // Reconstitute the covariance matrix with modified singular values using the column
    // // vectors in V.
    for (int k = 0; k < 3; k++) { // 为什么要这么做重构协方差矩阵
      Eigen::Vector3d col = U.col(k);
      double v = 1.; // biggest 2 singular values replaced by 1
      if (k == 2)    // smallest singular value replaced by gicp_epsilon
        v = gicp_epsilon_; // 表示拟合平面法向量的不确定性
      cov += v * col * col.transpose(); 
      // 前两个最大的奇异值被替换成1
      // 最小的奇异值被替换成gicp_epsilon
    }
  }
}

template <typename PointType>
bool GeneralizedICP<PointType>::align(const Matrix4d& initGuess, const PointCloudPtr aligned)
{
    // 基于initGuess配置参数块，参考转成t、q形式
    Eigen::Vector3d tmp_t = initGuess.block<3, 1>(0, 3);
    Eigen::Matrix3d tmp_r = initGuess.block<3, 3>(0, 0);
    Eigen::Quaterniond tmp_q(tmp_r);
    double parameters[7] = {tmp_q.x(), tmp_q.y(), tmp_q.z(), tmp_q.w(), tmp_t.x(), tmp_t.y(), tmp_t.z()};
    Eigen::Map<Eigen::Quaterniond> q(parameters); // 激光雷达在/map当前的位姿,Eigen::Map类似一个引用，Eigen::Map<Eigen::Quaterniond>新建一个四元数q_w_curr（矩阵），它的第一个参数由指针parameters给出，然后向后取4个数
    Eigen::Map<Eigen::Vector3d> t(parameters + 4);

    cost_last = DBL_MAX;
    for(int it=0; it<max_outer_iterations_; it++)
    {
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1); // 迭代的核函数
      ceres::Problem::Options problem_options;


      // 构建最小二乘问题
      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(parameters, 7, new PoseLocalParameterization());

      // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
      // 遍历source点云
      int counter = 0;
      vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VpointBs(tmp_src_->size());
      vector<int> selected(tmp_src_->size(), -1);
      #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
      for(int i=0; i<tmp_src_->size(); i++)
      {
        // 首先基于初始估计将s[i]变换到target系
        PointType pointA = tmp_src_->points[i];

        // 然后将T*s[i]放到target的kd tree中查找最近邻
        Vector3d VpointA{pointA.x, pointA.y, pointA.z};
        Vector3d T_pointA = q*VpointA + t;
        PointType Tpa;
        Tpa.x = T_pointA.x(); Tpa.y = T_pointA.y(); Tpa.z = T_pointA.z(); 
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeTarget->nearestKSearch(Tpa, 1, pointSearchInd, pointSearchSqDis);

        PointType pointB = tmp_tgt_->points[pointSearchInd[0]];

        if(pointSearchSqDis[0]>corr_dist_threshold_)
          continue;

        selected[i] = pointSearchInd[0];
        VpointBs[i]<<pointB.x, pointB.y, pointB.z;
      }

      for(int i=0; i<tmp_src_->size(); i++)
      {
        if(selected[i]==-1)
          continue;
        counter++;
        // 首先基于初始估计将s[i]变换到target系
        PointType pointA = tmp_src_->points[i];
        // 然后将T*s[i]放到target的kd tree中查找最近邻
        Vector3d VpointA{pointA.x, pointA.y, pointA.z};

        Eigen::Matrix3d M = (target_covariances_->at(selected[i]) + q.toRotationMatrix()*source_covariances_->at(i)*q.toRotationMatrix().transpose()).inverse();

        // ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
        //         PointPointFactor, 3, 7>(
        //       new PointPointFactor(VpointA, VpointBs[i], M));
        ceres::CostFunction *cost_function = new explictPointPointFactor(VpointA, VpointBs[i], M);
        problem.AddResidualBlock(cost_function, loss_function, parameters);
      }

      ceres::Solver::Options options;
      options.parameter_tolerance = inner_parameter_tolerance_;
      options.gradient_tolerance = inner_gradient_tolerace_;
      //配置增量方程的解法
      options.linear_solver_type = ceres::DENSE_QR;
      // 最大迭代次数
      options.max_num_iterations = max_inner_iterations_;
      // 不输出到cout
      options.minimizer_progress_to_stdout = false;
      // 检查由具有有限差分的每个残差块计算的所有雅可比行列式，比较结果，如果它们大不相同，
      // 则优化失败。如果设置为true比较耗费性能，一般保持默认false。
      options.check_gradients = false;
      
      options.num_threads = num_threads_;
      options.sparse_linear_algebra_library_type = ceres::NO_SPARSE;
      //优化信息
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary); // 求解

      // if(summary.IsSolutionUsable() && fabs(summary.final_cost/counter-cost_last)<optimization_epsilon_) // 内循环收敛即可退出外循环
      //   break;
      if(summary.IsSolutionUsable()) // 内循环收敛即可退出外循环
        break;

      cost_last = summary.final_cost/counter;
    }

    base_transformation_.setIdentity();
    base_transformation_.block<3, 3>(0, 0) = q.toRotationMatrix();
    base_transformation_.block<3, 1>(0, 3) = t;

    aligned->clear();
    pcl::transformPointCloud(*tmp_src_, *aligned, base_transformation_);
    return true;
}

template <typename PointType>
Matrix4d GeneralizedICP<PointType>::getTransformation()
{
    return base_transformation_;
}

template <typename PointType>
double GeneralizedICP<PointType>::getScore()
{
    return cost_last;
}

}

#endif