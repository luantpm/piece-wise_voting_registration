//
// Created by Luis A. Peralta M. on 20/07/01.
//
#pragma once

#include <map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

struct q_transformation_info {
    int votes{};
    std::vector<double> psi_v;
    std::vector<float> angles_v;
    std::vector<std::vector<int>> subsets_and_seed;
    std::vector<double> q_W_v;
    std::vector<double> q_X_v;
    std::vector<double> q_Y_v;
    std::vector<double> q_Z_v;
    std::vector<double> t_X_v;
    std::vector<double> t_Y_v;
    std::vector<double> t_Z_v;
};

struct ea_transformation_info {
    int votes{};
    std::vector<double> psi_v;
    std::vector<float> angles_v;
    std::vector<std::vector<int>> subsets_and_seed;
    std::vector<double> theta_X_v;
    std::vector<double> theta_Y_v;
    std::vector<double> theta_Z_v;
    std::vector<double> t_X_v;
    std::vector<double> t_Y_v;
    std::vector<double> t_Z_v;
};

using accumulator_7D = std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, q_transformation_info>>>>>>>;
using accumulator_6D = std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, ea_transformation_info>>>>>>;

using w_vector_double = std::vector<std::vector<double>>;
using w_vector_int = std::vector<std::vector<int>>;

using PointNormalsCloud = pcl::PointCloud<pcl::PointNormal>;
using NormalsCloud = pcl::PointCloud<pcl::Normal>;
using XYZLCloud = pcl::PointCloud<pcl::PointXYZL>;
using XYZCloud = pcl::PointCloud<pcl::PointXYZ>;


float ComputeCloudResolution(const XYZCloud::Ptr &cloud);


void ComputePointNormals(const XYZCloud::Ptr &cloud, float radius, const PointNormalsCloud::Ptr &normals_cloud);


void SupervoxelSegmentation(float voxel_resolution, float seed_radius, float color_weight, float spatial_weight, float normals_weight,
                            const XYZCloud::Ptr& cloud, XYZLCloud::Ptr& segmented_cloud, w_vector_int& subsets_points_ids);


void BuildSubsets(const XYZCloud::Ptr &cloud, const PointNormalsCloud::Ptr &normals_cloud, w_vector_int &points_subsets_ids_v,
                  std::vector<XYZCloud::Ptr> &points_subsets_v, std::vector<PointNormalsCloud::Ptr> &normals_subsets_v);


void BuildRotationMatrix(float angle, Eigen::Matrix4d &rotation_matrix, int base_axis);


void RunTranslation(const XYZCloud::Ptr &source, const XYZCloud::Ptr &target, Eigen::Matrix4d &transformation_matrix);


void RunICP(const XYZCloud::Ptr &source, const XYZCloud::Ptr &target, Eigen::Matrix4d &transformation_matrix, double &f_score);


void RunNICP(const PointNormalsCloud::Ptr &source_normals, const PointNormalsCloud::Ptr &target_normals, Eigen::Matrix4d &transformation_matrix,
             double &f_score);


void RunTrICP(const XYZCloud::Ptr &source, const XYZCloud::Ptr &target, Eigen::Matrix4d &transformation_matrix, double &f_score);


void RunLMICP(const XYZCloud::Ptr &source, const XYZCloud::Ptr &target, Eigen::Matrix4d &transformation_matrix, double &f_score);


void ComputeMetrics(const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, Eigen::Matrix4d &transformation_matrix, 
                    float inlier_threshold, float lambda, double &msep, double &psi);


void ComputeMetrics(const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, Eigen::Matrix4d &transformation_matrix, 
                    float inlier_threshold, float lambda, double &ovr, double &msep, double &psi);



void ComputeMetrics(const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, std::map<int, int> &source_density_scl_v, 
                    std::map<int, int> &target_density_scl_v, float inlier_threshold, float lambda, double &ovr, double &msep, double &psi, double &m_e);



void ComputeMetricsEulerAngles(const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, std::vector<double> &transformation_vector, 
                               int order, float inlier_threshold, float lambda, double &msep, double &psi);



void MatrixToVector(Eigen::Matrix4d &transformation_matrix, std::vector<double> &transformation_vector);



void MatrixToEulerAngles(Eigen::Matrix4d &transformation_matrix, std::vector<double> &transformation_vector, int order);



void VectorToMatrix(std::vector<double> &transformation_vector, Eigen::Matrix4d &transformation_matrix);



void EulerAnglesToMatrix(std::vector<double> &transformation_vector, Eigen::Matrix4d &transformation_matrix, int order);



void boundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointXYZ &minPoint, pcl::PointXYZ &maxPoint);



void Voting(std::vector<int> &subsets_i, int seed_radius, double psi, float angle,std::vector<double> &transformation_v,
            int bin_q_W, int bin_q_X, int bin_q_Y, int bin_q_Z, int bin_t_X, int bin_t_Y, int bin_t_Z, int v_stp, 
            accumulator_7D &accumulator, w_vector_int &aux_binning);



void Voting(std::vector<int> &subsets_i, int seed_radius, double psi, float angle, std::vector<double> &transformation_v, 
            int bin_theta_X, int bin_theta_Y, int bin_theta_Z, int bin_t_X, int bin_t_Y, int bin_t_Z, int v_stp, int order,
            accumulator_6D &accumulator, w_vector_int &aux_binning);



float GetFrequentAngle(std::vector<float>& angles_v);



void VotesCounting(accumulator_7D &accumulator, w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v,
                   std::vector<float> &angles_v);



void VotesCounting(accumulator_6D &accumulator, w_vector_int &aux_binning, int order, std::vector<int> &votes_v, 
                   std::vector<double> &psi_v, std::vector<float> &angles_v);



void PrintAccumulatorAndMetrics(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v, std::vector<float> &angles_v, 
                                std::stringstream &file_name);



void FindBestBinByVotes(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v, std::vector<int> &best_bin, 
                        std::ofstream& summary_file);



void FindBestBinByVotes(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v, std::vector<int> &best_bin, 
                        int &bin_id, int &votes);



void FindBestBinByPSI(w_vector_int &aux_binning, std::vector<double> &psi_v, std::vector<int> &best_bin, std::ofstream& summary_file);



void GetBestBinTransformation(accumulator_7D &accumulator, std::vector<int> &best_bin, w_vector_double &best_transformations_v, 
                              w_vector_int &best_corresponding_subsets_v, std::map<int, std::vector<int>> &seed_subsets_map);



void GetBestBinTransformation(accumulator_6D &accumulator, int order, std::vector<int> &best_bin, w_vector_double &best_transformations_v,
                              w_vector_int &best_corresponding_subsets_v, std::map<int, std::vector<int>> &seed_subsets_map);



void ComputeAverageTransformation(w_vector_double &transformations_v, std::vector<double> &av_transformation);



void ComputeAverageTransformationEA(w_vector_double &transformations_v, std::vector<double> &av_transformation);



void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* nothing);



void DynamicVisualizer(w_vector_double &transformations_v_a, std::vector<double> &transformation_v_b, const XYZCloud::Ptr &source_cloud, 
                       const XYZCloud::Ptr &target_cloud, std::vector<XYZCloud::Ptr> &corresponding_subsets_v);



void StaticVisualizer(Eigen::Matrix4d &tr_matrix, const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, float inlier_threshold);



void StaticVisualizer(const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud, float inlier_threshold);



void StaticVisualizerPOV(Eigen::Matrix4d &transformation_a, const XYZCloud::Ptr &source_cloud, const XYZCloud::Ptr &target_cloud,
                         float inlier_threshold, std::stringstream& file_base_name);


void Voxelization(size_t leaf_size, float resolution, const XYZCloud::Ptr &cloud, const pcl::PointXYZ &max_point,
                  w_vector_int &indices_v, std::vector<std::vector<Eigen::Vector3f>> &voxels_min_max_pt);



void HeatMapVisualizer(const XYZCloud::Ptr &cloud, w_vector_int &subsets_points_idx,
                       std::stringstream &file_base_name, std::ofstream &summary_file);



void HeatMapVisualizerPOV(const XYZCloud::Ptr &cloud, w_vector_int &subsets_points_idx,
                          std::stringstream &file_base_name, std::ofstream &summary_file);



void ComputeDensity(const XYZCloud::Ptr &cloud, std::map<int, double> &density_values, 
                    double &max_density, double &min_density, float radius);



double ComputeArea(const XYZCloud::Ptr &cloud, float radius);