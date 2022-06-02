//
// Created by Luis A. Peralta M. on 20/07/01.
//


#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>
#include <unordered_map>
#include <bits/stdc++.h>
#include <iostream>
#include <cassert>
#include <numeric>
#include <fstream>
#include <limits>
#include <ios>

#include <pcl/search/kdtree.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>

#include "tools.hpp"


bool close_visualizer = false;


float ComputeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    std::vector<int> nearest_neighbors;
    std::vector<float> nearest_neighbors_sqr_distances;
    int points = 0;
    float sqr_sum = 0.0;
    float resolution = 0.0;
    tree->setInputCloud(cloud);
    for (unsigned int i = 0; i < cloud->size(); ++i) {
        if (!isfinite((*cloud)[i].x))
            continue;
        tree->nearestKSearch(i, 2, nearest_neighbors, nearest_neighbors_sqr_distances);
        if (!nearest_neighbors.empty()) {
            sqr_sum += std::sqrt(nearest_neighbors_sqr_distances[1]);
            ++points;
        }
    }
    if (points != 0)
        resolution = sqr_sum / (float)points;
    return resolution;
}


void ComputePointNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float radius,
                         const pcl::PointCloud<pcl::PointNormal>::Ptr &normals_cloud) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    tree->setInputCloud(cloud);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.setViewPoint(0.0, 0.0, 0.0);
    ne.compute(*normals);
    concatenateFields(*cloud, *normals, *normals_cloud);
}


void SupervoxelSegmentation(float voxel_resolution, float seed_radius, float color_weight, float spatial_weight,
                            float normals_weight, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            pcl::PointCloud<pcl::PointXYZL>::Ptr &segmented_cloud, std::vector<std::vector<int>> &subsets_points_ids) {
    pcl::SupervoxelClustering<pcl::PointXYZ> svc (voxel_resolution, seed_radius);
    std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> clusters;
    std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> refined_clusters;
    svc.setInputCloud(cloud);
    svc.setColorImportance(color_weight);
    svc.setSpatialImportance(spatial_weight);
    svc.setNormalImportance(normals_weight);
    svc.extract(clusters);
    svc.refineSupervoxels(3, refined_clusters);
    segmented_cloud = svc.getLabeledVoxelCloud();
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    tree->setInputCloud(cloud);
    for (auto &cluster : refined_clusters) {
        std::vector<int> cluster_ids;
        for (auto point : cluster.second->voxels_->points) {
            std::vector<int> closest_points;
            std::vector<float> closest_points_sqr_distances;
            tree->nearestKSearch(point, 1, closest_points, closest_points_sqr_distances);
            if (!closest_points.empty()) {
                for (int closest_point : closest_points) {
                    auto it = find(cluster_ids.begin(), cluster_ids.end(), closest_point);
                    if (it != cluster_ids.end())
                        continue;
                    else
                        cluster_ids.push_back(closest_point);
                }
            }
        }
        subsets_points_ids.push_back(cluster_ids);
    }
}


void BuildSubsets(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointCloud<pcl::PointNormal>::Ptr &normals_cloud,
                  std::vector<std::vector<int>> &points_subsets_ids_v, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &points_subsets_v,
                  std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> &normals_subsets_v) {
    for (auto &i : points_subsets_ids_v) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_subset (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointNormal>::Ptr normals_subset (new pcl::PointCloud<pcl::PointNormal> ());
        points_subset->points.resize(i.size());
        normals_subset->points.resize(i.size());
        for (unsigned int j = 0; j < i.size(); ++j) {
            copyPoint(cloud->points[i[j]], points_subset->points[j]);
            copyPoint(normals_cloud->points[i[j]], normals_subset->points[j]);
        }
        points_subsets_v.push_back(points_subset);
        normals_subsets_v.push_back(normals_subset);
    }
}


void BuildRotationMatrix(float angle, Eigen::Matrix4d &rotation_matrix, int base_axis) {
    Eigen::Matrix3d rotation_angle_matrix;
    switch (base_axis) {
        case 1: // Rotate around X axis
            rotation_angle_matrix = Eigen::AngleAxisd(angle * (M_PI / 180.0), Eigen::Vector3d::UnitX());
        break;
        case 2: // Rotate around Y axis
            rotation_angle_matrix = Eigen::AngleAxisd(angle * (M_PI / 180.0), Eigen::Vector3d::UnitY());
        break;
        case 3: // Rotate around Z axis
            rotation_angle_matrix = Eigen::AngleAxisd(angle * (M_PI / 180.0), Eigen::Vector3d::UnitZ());
        break;
        default : // Rotate around Y axis by default
            rotation_angle_matrix = Eigen::AngleAxisd(angle * (M_PI / 180.0), Eigen::Vector3d::UnitY());
    }
    Eigen::Vector3d translation_null_vector(0.0, 0.0, 0.0);
    rotation_matrix.block(0, 0, 3, 3) = rotation_angle_matrix;
    rotation_matrix.block(0, 3, 3, 1) = translation_null_vector;
}


void RunTranslation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                    Eigen::Matrix4d &transformation_matrix) {
    pcl::PointXYZ source_centroid_point, target_centroid_point;
    pcl::CentroidPoint<pcl::PointXYZ> s_centroid, t_centroid;
    for (auto point : source->points)
        s_centroid.add(point);
    s_centroid.get(source_centroid_point);
    for (auto point : target->points)
        t_centroid.add(point);
    t_centroid.get(target_centroid_point);
    Eigen::Vector3d t_vector;
    t_vector(0) = target_centroid_point.x - source_centroid_point.x;
    t_vector(1) = target_centroid_point.y - source_centroid_point.y;
    t_vector(2) = target_centroid_point.z - source_centroid_point.z;
    transformation_matrix = Eigen::Matrix4d::Identity();
    transformation_matrix.block(0, 3, 3, 1) = t_vector;
}


void RunICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
            Eigen::Matrix4d &transformation_matrix, double &f_score) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_tr (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp (new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> ());
    icp->setInputSource(source);
    icp->setInputTarget(target);
    icp->align(*source_tr);
    if (icp->hasConverged()) {
        transformation_matrix = icp->getFinalTransformation().cast<double>();
        f_score = icp->getFitnessScore();
    } else {
        std::cout << "\t...registration did not converged." << std::endl;
        transformation_matrix = Eigen::Matrix4d::Identity();
        f_score = 100.0;
    }
}


void RunNICP(const pcl::PointCloud<pcl::PointNormal>::Ptr &source_normals, const pcl::PointCloud<pcl::PointNormal>::Ptr &target_normals,
             Eigen::Matrix4d &transformation_matrix, double &f_score){
    pcl::PointCloud<pcl::PointNormal>::Ptr source_tr (new pcl::PointCloud<pcl::PointNormal>());
    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>::Ptr nicp (new pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>());
    nicp->setInputSource(source_normals);
    nicp->setInputTarget(target_normals);
    nicp->align(*source_tr);
    if (nicp->hasConverged()) {
        transformation_matrix = nicp->getFinalTransformation().cast<double>();
        f_score = nicp->getFitnessScore();
    } else {
        std::cout << "\t...registration did not converged." << std::endl;
        transformation_matrix = Eigen::Matrix4d::Identity();
        f_score = 100.0;
    }
}


void RunTrICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
              Eigen::Matrix4d &transformation_matrix, double &f_score) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_tr (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr tricp (new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> ());
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr cor_rejector (new pcl::registration::CorrespondenceRejectorTrimmed ());
    cor_rejector->setOverlapRatio(0.9);
    tricp->addCorrespondenceRejector(cor_rejector);
    tricp->setInputSource(source);
    tricp->setInputTarget(target);
    tricp->align(*source_tr);
    if (tricp->hasConverged()) {
        transformation_matrix = tricp->getFinalTransformation().cast<double>();
        f_score = tricp->getFitnessScore();
    } else {
        std::cout << "\t...registration did not converged." << std::endl;
        transformation_matrix = Eigen::Matrix4d::Identity();
        f_score = 100.0;
    }
}


void RunLMICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
              Eigen::Matrix4d &transformation_matrix, double &f_score) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_tr (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ>::Ptr lmicp(new pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> ());
    lmicp->setInputSource(source);
    lmicp->setInputTarget(target);
    lmicp->align(*source_tr);
    if (lmicp->hasConverged()) {
        transformation_matrix = lmicp->getFinalTransformation().cast<double>();
        f_score = lmicp->getFitnessScore();
    } else {
        std::cout << "\t...registration did not converged." << std::endl;
        transformation_matrix = Eigen::Matrix4d::Identity();
        f_score = 100.0;
    }
}


void ComputeMetrics(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud,
                    Eigen::Matrix4d &transformation_matrix, float inlier_threshold, float lambda, double &msep, double &psi) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*source_cloud, *source_cloud_tr, transformation_matrix);
    int t_correspondences = 0;
    double penalty_sum = 0.0;
    double distance_sum = 0.0;
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    for (auto point : source_cloud_tr->points) {
        std::vector<int> nn (1);
        std::vector<float> nn_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nn, nn_sqr_distance);
        if (!nn.empty()) {
            t_correspondences++;
            distance_sum += std::sqrt(nn_sqr_distance[0]);
        } else
            penalty_sum += 1.0;
    }
    double ovr = (((double) t_correspondences) / (double)source_cloud->points.size()) * 100;
    msep = (distance_sum + penalty_sum) / (double)source_cloud->points.size();
    psi = (msep / pow(ovr, 1 + lambda));
}

void ComputeMetrics(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud,
                    Eigen::Matrix4d &transformation_matrix, float inlier_threshold, float lambda,
                    double &ovr, double &msep, double &psi) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*source_cloud, *source_cloud_tr, transformation_matrix);
    int t_correspondences = 0;
    double penalty_sum = 0.0;
    double distance_sum = 0.0;
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    for (auto point : source_cloud_tr->points) {
        std::vector<int> nn (1);
        std::vector<float> nn_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nn, nn_sqr_distance);
        if (!nn.empty()) {
            t_correspondences++;
            distance_sum += std::sqrt(nn_sqr_distance[0]);
        } else
            penalty_sum += 1.0;
    }
    ovr = (((double) t_correspondences) / (double)source_cloud->points.size()) * 100;
    msep = (distance_sum + penalty_sum) / (double)source_cloud->points.size();
    psi = (msep / pow(ovr, 1 + lambda));
}

void ComputeMetrics(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud,
                    std::map<int, int> &source_density_scl_v, std::map<int, int> &target_density_scl_v, float inlier_threshold,
                    float lambda, double &ovr, double &msep, double &psi, double &m_e) {
    int t_correspondences = 0;
    double penalty_sum = 0.0;
    double distance_sum = 0.0;
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    for (unsigned int p_i = 0; p_i < source_cloud->points.size(); ++p_i) {
        int d_diff;
        float weight;
        std::vector<int> nn (1);
        std::vector<float> nn_sqr_distance (1);
        tree.radiusSearch(source_cloud->points[p_i], inlier_threshold, nn, nn_sqr_distance);
        if (!nn.empty()) {
            // There is a point within the radius
            t_correspondences++;
            // Check the density scale difference between the points
            d_diff = abs(source_density_scl_v[p_i] - target_density_scl_v[nn[0]]);
            // Define the weight
            if (d_diff == 1)
                weight = 1.0;
            else if (d_diff == 0)
                weight = 3.0;
            else
                weight = 2.0;
            // Sum the distances
            distance_sum += std::sqrt(nn_sqr_distance[0]) * weight;
        } else
            penalty_sum += 1.0;
    }
    ovr = ((double) t_correspondences / (double)source_cloud->points.size()) * 100;
    msep = (distance_sum + penalty_sum) / (double)source_cloud->points.size();
    psi = (msep / pow(ovr, 1 + lambda));
    m_e = distance_sum;
}

void ComputeMetricsEulerAngles(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud, std::vector<double> &transformation_vector,
                               int order, float inlier_threshold, float lambda, double &msep, double &psi) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr (new pcl::PointCloud<pcl::PointXYZ>());
    double theta_X = transformation_vector[0];
    double theta_Y = transformation_vector[1];
    double theta_Z = transformation_vector[2];
    // Apply the translation first
    Eigen::Matrix4d translation_only_matrix = Eigen::Matrix4d::Identity();
    Eigen::Vector3d translation_only_vector;
    translation_only_vector(0) = transformation_vector[3];
    translation_only_vector(1) = transformation_vector[4];
    translation_only_vector(2) = transformation_vector[5];
    translation_only_matrix.block<3, 1>(0, 3) = translation_only_vector;
    transformPointCloud(*source_cloud, *source_cloud_tr, translation_only_matrix);
    // Rotate the source cloud according to Euler Angles' order
    Eigen::Matrix4d rotation_only_matrix;
    Eigen::Matrix3d rotation_X, rotation_Y, rotation_Z;
    rotation_X = Eigen::AngleAxisd(theta_X, Eigen::Vector3d::UnitX());
    rotation_Y = Eigen::AngleAxisd(theta_Y, Eigen::Vector3d::UnitY());
    rotation_Z = Eigen::AngleAxisd(theta_Z, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d translation_null_vector(0.0, 0.0, 0.0);
    rotation_only_matrix.block<3, 1>(0, 3) = translation_null_vector;
    switch (order) {
        case 1 :    // XYZ order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_Z * rotation_Y * rotation_X;
            break;
        case 2 :    // XZY order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_Y * rotation_Z * rotation_X;
            break;
        case 3 :    // YXZ order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_Z * rotation_X * rotation_Y;
            break;
        case 4:     // YZX order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_X * rotation_Z * rotation_Y;
            break;
        case 5:     // ZXY order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_Y * rotation_X * rotation_Z;
            break;
        case 6:     // ZYX order
            rotation_only_matrix.block<3, 3>(0, 0) = rotation_X * rotation_Y * rotation_Z;
            break;
        default:
            assert(false);
    }
    transformPointCloud(*source_cloud_tr, *source_cloud_tr, rotation_only_matrix);
    int t_correspondences = 0;
    double penalty_sum = 0.0;
    double distance_sum = 0.0;
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    for (auto point : source_cloud_tr->points) {
        std::vector<int> nn (1);
        std::vector<float> nn_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nn, nn_sqr_distance);
        if (!nn.empty()) {
            t_correspondences++;
            distance_sum += std::sqrt(nn_sqr_distance[0]);
        } else
            penalty_sum += 1.0;
    }
    double ovr = (((double) t_correspondences) / (double)source_cloud->points.size()) * 100;
    msep = (distance_sum + penalty_sum) / (double)source_cloud->points.size();
    psi = (msep / pow(ovr, 1 + lambda));
}

void MatrixToVector(Eigen::Matrix4d &transformation_matrix, std::vector<double> &transformation_vector) {
    Eigen::Matrix3d r_matrix = transformation_matrix.block(0, 0, 3, 3);
    Eigen::Quaterniond r_quaternion(r_matrix);
    Eigen::Vector3d t_vector(transformation_matrix.block(0, 3, 3, 1));
    transformation_vector.resize(7);
    transformation_vector[0] = r_quaternion.w();
    transformation_vector[1] = r_quaternion.x();
    transformation_vector[2] = r_quaternion.y();
    transformation_vector[3] = r_quaternion.z();
    transformation_vector[4] = t_vector(0);
    transformation_vector[5] = t_vector(1);
    transformation_vector[6] = t_vector(2);
}


void MatrixToEulerAngles(Eigen::Matrix4d &transformation_matrix, std::vector<double> &transformation_vector, int order) {
    Eigen::Matrix3d r_matrix = transformation_matrix.block(0, 0, 3, 3);
    Eigen::Vector3d t_vector(transformation_matrix.block(0, 3, 3, 1));
    transformation_vector.resize(6);
    // Rotation matrix elements for simplification.
    double r_00 = r_matrix.coeff(0, 0);
    double r_01 = r_matrix.coeff(0, 1);
    double r_02 = r_matrix.coeff(0, 2);
    double r_10 = r_matrix.coeff(1, 0);
    double r_11 = r_matrix.coeff(1, 1);
    double r_12 = r_matrix.coeff(1, 2);
    double r_20 = r_matrix.coeff(2, 0);
    double r_21 = r_matrix.coeff(2, 1);
    double r_22 = r_matrix.coeff(2, 2);
    double theta_X = 0.0;
    double theta_Y = 0.0;
    double theta_Z = 0.0;
    // Factoring according to chosen order.
    switch (order) {
        case 1: {   // XYZ order
            if (r_02 < 1) {
                if (r_02 > -1) {
                    theta_Y = asin(r_02);
                    theta_X = atan2(-r_12, r_22);
                    theta_Z = atan2(-r_01, r_00);
                } else {    // r_02 = -1
                    // Not a unique solution: theta_Z - theta_X = atan2(r_10, r_11)
                    theta_Y = -M_PI / 2;
                    theta_X = -atan2(r_10, r_11);
                    theta_Z = 0;
                }
            } else {    // r_02 = +1
                // Not a unique solution: theta_Z + theta_X = atan2(r_10, r_11)
                theta_Y = M_PI / 2;
                theta_X = atan2(r_10, r_11);
                theta_Z = 0;
            }
            break;
        }
        case 2: {   // XZY order
            if (r_01 < 1) {
                if (r_01 > -1) {
                    theta_Z = asin(-r_01);
                    theta_X = atan2(r_21, r_11);
                    theta_Y = atan2(r_02, r_00);
                } else {    // r_01 = -1
                    // Not a unique solution: theta_Y - theta_X = atan2(-r_20, r_22)
                    theta_Z = M_PI / 2;
                    theta_X = -atan2(-r_20, r_22);
                    theta_Y = 0;
                }
            } else {    // r_01 = 1
                // Not a unique solution: theta_Y + theta_X = atan2(-r_20, r_22)
                theta_Z = -M_PI / 2;
                theta_X = atan2(-r_20, r_22);
                theta_Y = 0;
            }
            break;
        }
        case 3: {   // YXZ order
            if (r_12 < 1) {
                if (r_12 > -1) {
                    theta_X = asin(-r_12);
                    theta_Y = atan2(r_02, r_22);
                    theta_Z = atan2(r_10, r_11);
                } else {    // r_12 = -1
                    // Not a unique solution: theta_Z - theta_Y = atan2(-r_01, r_00)
                    theta_X = M_PI / 2;
                    theta_Y = -atan2(-r_01, r_00);
                    theta_Z = 0;
                }
            } else {    // r_12 = 1
                // Not a unique solution: theta_Z + theta_Y = atan2(-r_01, r_00)
                theta_X = -M_PI / 2;
                theta_Y = atan2(-r_01, r_00);
                theta_Z = 0;
            }
            break;
        }
        case 4: {   // YZX order
            if (r_10 < 1) {
                if (r_10 > -1) {
                    theta_Z = asin(r_10);
                    theta_Y = atan2(-r_20, r_00);
                    theta_X = atan2(-r_12, r_11);
                } else {    // r_10 = -1
                    // Not a unique solution: theta_X - theta_Y = atan2(r_21, r_22)
                    theta_Z = -M_PI / 2;
                    theta_Y = -atan2(r_21, r_22);
                    theta_X = 0;
                }
            } else {    // r_10 = 1
                // Not a unique solution: theta_X + theta_Y = atan2(r_21, r_22)
                theta_Z = M_PI / 2;
                theta_Y = atan2(r_21, r_22);
                theta_X = 0;
            }
            break;
        }
        case 5: {   // ZXY order
            if (r_21 < 1) {
                if (r_21 > -1) {
                    theta_X = asin(r_21);
                    theta_Z = atan2(-r_01, r_11);
                    theta_Y = atan2(-r_20, r_22);
                } else {    // r_21 = -1
                    // Not a unique solution: theta_Y - theta_Z = atan2(r_02, r_00)
                    theta_X = -M_PI / 2;
                    theta_Z = -atan2(r_02, r_00);
                    theta_Y = 0;
                }
            } else {    // r_21 = 1
                // Not a unique solution: theta_Y + theta_Z = atan2(r_02, r_00)
                theta_X = M_PI / 2;
                theta_Z = atan2(r_02, r_00);
                theta_Y = 0;
            }
            break;
        }
        case 6: {   // ZYX order
            if (r_20 < 1) {
                if (r_20 > -1) {
                    theta_Y = asin(-r_20);
                    theta_Z = atan2(r_10, r_00);
                    theta_X = atan2(r_21, r_22);
                } else {    // r_20 = -1
                    // Not a unique solution: theta_X - theta_Z =  atan2(-r_12, r_11)
                    theta_Y = M_PI / 2;
                    theta_Z = -atan2(-r_12, r_11);
                    theta_X = 0;
                }
            } else {    // r_20 = 1
                // Not a unique solution: theta_X + theta_Z = atan2(-r_12, r_11)
                theta_Y = -M_PI / 2;
                theta_Z = atan2(-r_12, r_11);
                theta_X = 0;
            }
            break;
        }
        default:
            assert(false);
            break;
    }
    transformation_vector[0] = theta_X;
    transformation_vector[1] = theta_Y;
    transformation_vector[2] = theta_Z;
    transformation_vector[3] = t_vector(0);
    transformation_vector[4] = t_vector(1);
    transformation_vector[5] = t_vector(2);
}


void VectorToMatrix(std::vector<double> &transformation_vector, Eigen::Matrix4d &transformation_matrix) {
    Eigen::Quaterniond r_quaternion;
    r_quaternion.w() = transformation_vector[0];
    r_quaternion.x() = transformation_vector[1];
    r_quaternion.y() = transformation_vector[2];
    r_quaternion.z() = transformation_vector[3];
    Eigen::Vector3d t_vector;
    t_vector(0) = transformation_vector[4];
    t_vector(1) = transformation_vector[5];
    t_vector(2) = transformation_vector[6];
    Eigen::Matrix3d r_matrix(r_quaternion);
    transformation_matrix.block(0, 0, 3, 3) = r_matrix;
    transformation_matrix.block(0, 3, 3, 1) = t_vector;
}


void EulerAnglesToMatrix(std::vector<double> &transformation_vector, Eigen::Matrix4d &transformation_matrix, int order) {
    double theta_X = transformation_vector[0];
    double theta_Y = transformation_vector[1];
    double theta_Z = transformation_vector[2];
    Eigen::Matrix3d rotation_X, rotation_Y, rotation_Z;
    rotation_X = Eigen::AngleAxisd(theta_X, Eigen::Vector3d::UnitX());
    rotation_Y = Eigen::AngleAxisd(theta_Y, Eigen::Vector3d::UnitY());
    rotation_Z = Eigen::AngleAxisd(theta_Z, Eigen::Vector3d::UnitZ());
    switch (order) {
        case 1:   // XYZ oder
            transformation_matrix.block<3, 3>(0, 0) = rotation_Z * rotation_Y * rotation_X;
            break;
        case 2:   // XZY order
            transformation_matrix.block<3, 3>(0, 0) = rotation_Y * rotation_Z * rotation_X;
            break;
        case 3:   // YXZ order
            transformation_matrix.block<3, 3>(0, 0) = rotation_Z * rotation_X * rotation_Y;
            break;
        case 4:   // YZX order
            transformation_matrix.block<3, 3>(0, 0) = rotation_X * rotation_Z * rotation_Y;
            break;
        case 5:   // ZXY order
            transformation_matrix.block<3, 3>(0, 0) = rotation_Y * rotation_X * rotation_Z;
            break;
        case 6:   // ZYX order
            transformation_matrix.block<3, 3>(0, 0) = rotation_X * rotation_Y * rotation_Z;
            break;
        default:
            assert(false);
    }
    Eigen::Vector3d t_vector;
    t_vector(0) = transformation_vector[3];
    t_vector(1) = transformation_vector[4];
    t_vector(2) = transformation_vector[5];
    transformation_matrix.block<3, 1>(0, 3) = t_vector;
}


void boundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointXYZ &minPoint, pcl::PointXYZ &maxPoint) {
    Eigen::Vector4f pca_centroid;
    Eigen::Matrix3f cov_matrix;
    pcl::compute3DCentroid(*cloud, pca_centroid);
    pcl::computeCovarianceMatrixNormalized(*cloud, pca_centroid, cov_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver (cov_matrix, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
    eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
    Eigen::Matrix4f prj_transformation (Eigen::Matrix4f::Identity());
    prj_transformation.block<3, 3>(0, 0) = eigen_vectors.transpose();
    prj_transformation.block<3, 1>(0, 3) = -1.0f * (prj_transformation.block<3, 3>(0, 0) * pca_centroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr prj_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud(*cloud, *prj_cloud, prj_transformation);
    pcl::getMinMax3D(*prj_cloud, minPoint, maxPoint);
}


void Voting(std::vector<int> &subsets_i, int seed_radius, double psi, float angle,std::vector<double> &transformation_v,
            int bin_q_W, int bin_q_X, int bin_q_Y, int bin_q_Z, int bin_t_X, int bin_t_Y, int bin_t_Z, int v_stp,
            accumulator_7D &accumulator, w_vector_int &aux_binning) {
    // Look  for quaternion W
    auto q_W_it = accumulator.find(bin_q_W);
    if (q_W_it != accumulator.end()) {  // Look for quaternion X
        auto q_X_it = q_W_it->second.find(bin_q_X);
        if (q_X_it != q_W_it->second.end()) {   // Look for quaternion Y
            auto q_Y_it = q_X_it->second.find(bin_q_Y);
            if (q_Y_it != q_X_it->second.end()) {   // Look for quaternion Z
                auto q_Z_it = q_Y_it->second.find(bin_q_Z);
                if (q_Z_it != q_Y_it->second.end()) {   // Look for translation X
                    auto t_X_it = q_Z_it->second.find(bin_t_X);
                    if (t_X_it != q_Z_it->second.end()) {   // Look for translation Y
                        auto t_Y_it = t_X_it->second.find(bin_t_Y);
                        if (t_Y_it != t_X_it->second.end()) {   // Look for translation Z
                            auto t_Z_it = t_Y_it->second.find(bin_t_Z);
                            if (t_Z_it != t_Y_it->second.end()) {   // The full transformation was found.
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_Z_it->second.q_W_v.push_back(transformation_v[0]);
                                t_Z_it->second.q_X_v.push_back(transformation_v[1]);
                                t_Z_it->second.q_Y_v.push_back(transformation_v[2]);
                                t_Z_it->second.q_Z_v.push_back(transformation_v[3]);
                                t_Z_it->second.t_X_v.push_back(transformation_v[4]);
                                t_Z_it->second.t_Y_v.push_back(transformation_v[5]);
                                t_Z_it->second.t_Z_v.push_back(transformation_v[6]);
                                t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                t_Z_it->second.angles_v.push_back(angle);
                                t_Z_it->second.psi_v.push_back(psi);
                                t_Z_it->second.votes += v_stp;
                            } else {    // Translation Z was not found. Thus, it is a new transformation
                                q_transformation_info t_i = q_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.q_W_v.clear();
                                t_i.q_X_v.clear();
                                t_i.q_Y_v.clear();
                                t_i.q_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give it one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.q_W_v.push_back(transformation_v[0]);
                                t_i.q_X_v.push_back(transformation_v[1]);
                                t_i.q_Y_v.push_back(transformation_v[2]);
                                t_i.q_Z_v.push_back(transformation_v[3]);
                                t_i.t_X_v.push_back(transformation_v[4]);
                                t_i.t_Y_v.push_back(transformation_v[5]);
                                t_i.t_Z_v.push_back(transformation_v[6]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_Y_it->second[bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation Y was not found. Thus, it is a new transformation
                            q_transformation_info t_i = q_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.q_W_v.clear();
                            t_i.q_X_v.clear();
                            t_i.q_Y_v.clear();
                            t_i.q_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.q_W_v.push_back(transformation_v[0]);
                            t_i.q_X_v.push_back(transformation_v[1]);
                            t_i.q_Y_v.push_back(transformation_v[2]);
                            t_i.q_Z_v.push_back(transformation_v[3]);
                            t_i.t_X_v.push_back(transformation_v[4]);
                            t_i.t_Y_v.push_back(transformation_v[5]);
                            t_i.t_Z_v.push_back(transformation_v[6]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Translation X was not found. Thus, it is a new transformation
                        q_transformation_info t_i = q_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.q_W_v.clear();
                        t_i.q_X_v.clear();
                        t_i.q_Y_v.clear();
                        t_i.q_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.q_W_v.push_back(transformation_v[0]);
                        t_i.q_X_v.push_back(transformation_v[1]);
                        t_i.q_Y_v.push_back(transformation_v[2]);
                        t_i.q_Z_v.push_back(transformation_v[3]);
                        t_i.t_X_v.push_back(transformation_v[4]);
                        t_i.t_Y_v.push_back(transformation_v[5]);
                        t_i.t_Z_v.push_back(transformation_v[6]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        q_Z_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Quaternion Z was not found. Thus, it is a new transformation
                    q_transformation_info t_i = q_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.q_W_v.clear();
                    t_i.q_X_v.clear();
                    t_i.q_Y_v.clear();
                    t_i.q_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.q_W_v.push_back(transformation_v[0]);
                    t_i.q_X_v.push_back(transformation_v[1]);
                    t_i.q_Y_v.push_back(transformation_v[2]);
                    t_i.q_Z_v.push_back(transformation_v[3]);
                    t_i.t_X_v.push_back(transformation_v[4]);
                    t_i.t_Y_v.push_back(transformation_v[5]);
                    t_i.t_Z_v.push_back(transformation_v[6]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    q_Y_it->second[bin_q_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Quaternion Y was not found. Thus, it is a new transformation.
                q_transformation_info t_i = q_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.q_W_v.clear();
                t_i.q_X_v.clear();
                t_i.q_Y_v.clear();
                t_i.q_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.q_W_v.push_back(transformation_v[0]);
                t_i.q_X_v.push_back(transformation_v[1]);
                t_i.q_Y_v.push_back(transformation_v[2]);
                t_i.q_Z_v.push_back(transformation_v[3]);
                t_i.t_X_v.push_back(transformation_v[4]);
                t_i.t_Y_v.push_back(transformation_v[5]);
                t_i.t_Z_v.push_back(transformation_v[6]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                q_X_it->second[bin_q_Y][bin_q_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
        } else {    // Quaternion X was not found. Thus, it is a new transformation
            q_transformation_info t_i = q_transformation_info();
            // Initialize all containers as empty and zero votes
            t_i.q_W_v.clear();
            t_i.q_X_v.clear();
            t_i.q_Y_v.clear();
            t_i.q_Z_v.clear();
            t_i.t_X_v.clear();
            t_i.t_Y_v.clear();
            t_i.t_Z_v.clear();
            t_i.subsets_and_seed.clear();
            t_i.angles_v.clear();
            t_i.psi_v.clear();
            t_i.votes = 0;
            // Add the information of the very first transformation and give one vote
            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
            t_i.q_W_v.push_back(transformation_v[0]);
            t_i.q_X_v.push_back(transformation_v[1]);
            t_i.q_Y_v.push_back(transformation_v[2]);
            t_i.q_Z_v.push_back(transformation_v[3]);
            t_i.t_X_v.push_back(transformation_v[4]);
            t_i.t_Y_v.push_back(transformation_v[5]);
            t_i.t_Z_v.push_back(transformation_v[6]);
            t_i.subsets_and_seed.push_back(ss_sr);
            t_i.angles_v.push_back(angle);
            t_i.psi_v.push_back(psi);
            t_i.votes = v_stp;
            q_W_it->second[bin_q_X][bin_q_Y][bin_q_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
            // Add the bin to the aux accumulator
            std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
            aux_binning.push_back(binning);
        }
    } else {    // Quaternion Q was not found. Thus, it is a new transformation
        q_transformation_info t_i = q_transformation_info();
        // Initialize all containers as empty and zero votes
        t_i.q_W_v.clear();
        t_i.q_X_v.clear();
        t_i.q_Y_v.clear();
        t_i.q_Z_v.clear();
        t_i.t_X_v.clear();
        t_i.t_Y_v.clear();
        t_i.t_Z_v.clear();
        t_i.subsets_and_seed.clear();
        t_i.angles_v.clear();
        t_i.psi_v.clear();
        t_i.votes = 0;
        // Add the information of the very first transformation and give one vote
        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
        t_i.q_W_v.push_back(transformation_v[0]);
        t_i.q_X_v.push_back(transformation_v[1]);
        t_i.q_Y_v.push_back(transformation_v[2]);
        t_i.q_Z_v.push_back(transformation_v[3]);
        t_i.t_X_v.push_back(transformation_v[4]);
        t_i.t_Y_v.push_back(transformation_v[5]);
        t_i.t_Z_v.push_back(transformation_v[6]);
        t_i.subsets_and_seed.push_back(ss_sr);
        t_i.angles_v.push_back(angle);
        t_i.psi_v.push_back(psi);
        t_i.votes = v_stp;
        accumulator[bin_q_W][bin_q_X][bin_q_Y][bin_q_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
        // Add the bin to the aux accumulator
        std::vector<int> binning {bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z};
        aux_binning.push_back(binning);
    }
}


void Voting(std::vector<int> &subsets_i, int seed_radius, double psi, float angle, std::vector<double> &transformation_v,
            int bin_theta_X, int bin_theta_Y, int bin_theta_Z, int bin_t_X, int bin_t_Y, int bin_t_Z, int v_stp,
            int order, accumulator_6D &accumulator, w_vector_int &aux_binning) {
    switch (order) {
        case 1: {   // XYZ order
            // Look  for theta_X
            auto theta_X_it = accumulator.find(bin_theta_X);
            if (theta_X_it != accumulator.end()) {
                // Look for theta_Y
                auto theta_Y_it = theta_X_it->second.find(bin_theta_Y);
                if (theta_Y_it != theta_X_it->second.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_Y_it->second.find(bin_theta_Z);
                    if (theta_Z_it != theta_Y_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_Z_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_Z_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_Z_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_Z was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_Y_it->second[bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_Y was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_X_it->second[bin_theta_Y][bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_X was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_X][bin_theta_Y][bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_theta_X, bin_theta_Y, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        case 2: {   // XZY order
            // Look  for theta_X
            auto theta_X_it = accumulator.find(bin_theta_X);
            if (theta_X_it != accumulator.end()) {
                // Look for theta_Z
                auto theta_Z_it = theta_X_it->second.find(bin_theta_Z);
                if (theta_Z_it != theta_X_it->second.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_Z_it->second.find(bin_theta_Y);
                    if (theta_Y_it != theta_Z_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_Y_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_Y_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_Y_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_Y was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_Z_it->second[bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_Z was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_X_it->second[bin_theta_Z][bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_X was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr{subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_X][bin_theta_Z][bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning{bin_theta_X, bin_theta_Z, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        case 3: {   // YXZ order
            // Look  for theta_Y
            auto theta_Y_it = accumulator.find(bin_theta_Y);
            if (theta_Y_it != accumulator.end()) {
                // Look for theta_X
                auto theta_X_it = theta_Y_it->second.find(bin_theta_X);
                if (theta_X_it != theta_Y_it->second.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_X_it->second.find(bin_theta_Z);
                    if (theta_Z_it != theta_X_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_Z_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_Z_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_Y, bin_theta_X, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_Y, bin_theta_X, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_Z_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_Z was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_X_it->second[bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_Y, bin_theta_X, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_X was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_Y_it->second[bin_theta_X][bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_Y, bin_theta_X, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_Y was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_Y][bin_theta_X][bin_theta_Z][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_theta_Y, bin_theta_X, bin_theta_Z, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        case 4: {   // YZX order
            // Look  for theta_Y
            auto theta_Y_it = accumulator.find(bin_theta_Y);
            if (theta_Y_it != accumulator.end()) {
                // Look for theta_Z
                auto theta_Z_it = theta_Y_it->second.find(bin_theta_Z);
                if (theta_Z_it != theta_Y_it->second.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Z_it->second.find(bin_theta_X);
                    if (theta_X_it != theta_Z_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_X_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_X_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_X_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_X was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_Z_it->second[bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_Z was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_Y_it->second[bin_theta_Z][bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_Y was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_Y][bin_theta_Z][bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_theta_Y, bin_theta_Z, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        case 5: {   // ZXY order
            // Look  for theta_Z
            auto theta_Z_it = accumulator.find(bin_theta_Z);
            if (theta_Z_it != accumulator.end()) {
                // Look for theta_X
                auto theta_X_it = theta_Z_it->second.find(bin_theta_X);
                if (theta_X_it != theta_Z_it->second.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_X_it->second.find(bin_theta_Y);
                    if (theta_Y_it != theta_X_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_Y_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_Y_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_Y_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_Y was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_X_it->second[bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_X was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_Z_it->second[bin_theta_X][bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_Z was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_Z][bin_theta_X][bin_theta_Y][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_theta_Z, bin_theta_X, bin_theta_Y, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        case 6: {   // ZYX order
            // Look  for theta_Z
            auto theta_Z_it = accumulator.find(bin_theta_Z);
            if (theta_Z_it != accumulator.end()) {
                // Look for theta_Y
                auto theta_Y_it = theta_Z_it->second.find(bin_theta_Y);
                if (theta_Y_it != theta_Z_it->second.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Y_it->second.find(bin_theta_X);
                    if (theta_X_it != theta_Y_it->second.end()) {
                        // Look for translation X
                        auto t_X_it = theta_X_it->second.find(bin_t_X);
                        // Look for translation Y
                        if (t_X_it != theta_X_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(bin_t_X);
                            // Look for translation Z
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(bin_t_Y);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // The full transformation was found.
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_Z_it->second.theta_X_v.push_back(transformation_v[0]);
                                    t_Z_it->second.theta_Y_v.push_back(transformation_v[1]);
                                    t_Z_it->second.theta_Z_v.push_back(transformation_v[2]);
                                    t_Z_it->second.t_X_v.push_back(transformation_v[3]);
                                    t_Z_it->second.t_Y_v.push_back(transformation_v[4]);
                                    t_Z_it->second.t_Z_v.push_back(transformation_v[5]);
                                    t_Z_it->second.subsets_and_seed.push_back(ss_sr);
                                    t_Z_it->second.angles_v.push_back(angle);
                                    t_Z_it->second.psi_v.push_back(psi);
                                    t_Z_it->second.votes += v_stp;
                                } else {    // Translation Z was not found. Thus, it is a new transformation
                                    ea_transformation_info t_i = ea_transformation_info();
                                    // Initialize all containers as empty and zero votes
                                    t_i.theta_X_v.clear();
                                    t_i.theta_Y_v.clear();
                                    t_i.theta_Z_v.clear();
                                    t_i.t_X_v.clear();
                                    t_i.t_Y_v.clear();
                                    t_i.t_Z_v.clear();
                                    t_i.subsets_and_seed.clear();
                                    t_i.angles_v.clear();
                                    t_i.psi_v.clear();
                                    t_i.votes = 0;
                                    // Add the information of the very first transformation and give it a Voting step
                                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                    t_i.theta_X_v.push_back(transformation_v[0]);
                                    t_i.theta_Y_v.push_back(transformation_v[1]);
                                    t_i.theta_Z_v.push_back(transformation_v[2]);
                                    t_i.t_X_v.push_back(transformation_v[3]);
                                    t_i.t_Y_v.push_back(transformation_v[4]);
                                    t_i.t_Z_v.push_back(transformation_v[5]);
                                    t_i.subsets_and_seed.push_back(ss_sr);
                                    t_i.angles_v.push_back(angle);
                                    t_i.psi_v.push_back(psi);
                                    t_i.votes = v_stp;
                                    t_Y_it->second[bin_t_Z] = t_i;
                                    // Add the bin to the aux accumulator
                                    std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                                    aux_binning.push_back(binning);
                                }
                            } else {    // Translation Y was not found. Thus, it is a new transformation
                                ea_transformation_info t_i = ea_transformation_info();
                                // Initialize all containers as empty and zero votes
                                t_i.theta_X_v.clear();
                                t_i.theta_Y_v.clear();
                                t_i.theta_Z_v.clear();
                                t_i.t_X_v.clear();
                                t_i.t_Y_v.clear();
                                t_i.t_Z_v.clear();
                                t_i.subsets_and_seed.clear();
                                t_i.angles_v.clear();
                                t_i.psi_v.clear();
                                t_i.votes = 0;
                                // Add the information of the very first transformation and give one vote
                                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                                t_i.theta_X_v.push_back(transformation_v[0]);
                                t_i.theta_Y_v.push_back(transformation_v[1]);
                                t_i.theta_Z_v.push_back(transformation_v[2]);
                                t_i.t_X_v.push_back(transformation_v[3]);
                                t_i.t_Y_v.push_back(transformation_v[4]);
                                t_i.t_Z_v.push_back(transformation_v[5]);
                                t_i.subsets_and_seed.push_back(ss_sr);
                                t_i.angles_v.push_back(angle);
                                t_i.psi_v.push_back(psi);
                                t_i.votes = v_stp;
                                t_X_it->second[bin_t_Y][bin_t_Z] = t_i;
                                // Add the bin to the aux accumulator
                                std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                                aux_binning.push_back(binning);
                            }
                        } else {    // Translation X was not found. Thus, it is a new transformation
                            ea_transformation_info t_i = ea_transformation_info();
                            // Initialize all containers as empty and zero votes
                            t_i.theta_X_v.clear();
                            t_i.theta_Y_v.clear();
                            t_i.theta_Z_v.clear();
                            t_i.t_X_v.clear();
                            t_i.t_Y_v.clear();
                            t_i.t_Z_v.clear();
                            t_i.subsets_and_seed.clear();
                            t_i.angles_v.clear();
                            t_i.psi_v.clear();
                            t_i.votes = 0;
                            // Add the information of the very first transformation and give one vote
                            std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                            t_i.theta_X_v.push_back(transformation_v[0]);
                            t_i.theta_Y_v.push_back(transformation_v[1]);
                            t_i.theta_Z_v.push_back(transformation_v[2]);
                            t_i.t_X_v.push_back(transformation_v[3]);
                            t_i.t_Y_v.push_back(transformation_v[4]);
                            t_i.t_Z_v.push_back(transformation_v[5]);
                            t_i.subsets_and_seed.push_back(ss_sr);
                            t_i.angles_v.push_back(angle);
                            t_i.psi_v.push_back(psi);
                            t_i.votes = v_stp;
                            theta_X_it->second[bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                            // Add the bin to the aux accumulator
                            std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                            aux_binning.push_back(binning);
                        }
                    } else {    // Angle theta_X was not found. Thus, it is a new transformation.
                        ea_transformation_info t_i = ea_transformation_info();
                        // Initialize all containers as empty and zero votes
                        t_i.theta_X_v.clear();
                        t_i.theta_Y_v.clear();
                        t_i.theta_Z_v.clear();
                        t_i.t_X_v.clear();
                        t_i.t_Y_v.clear();
                        t_i.t_Z_v.clear();
                        t_i.subsets_and_seed.clear();
                        t_i.angles_v.clear();
                        t_i.psi_v.clear();
                        t_i.votes = 0;
                        // Add the information of the very first transformation and give one vote
                        std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                        t_i.theta_X_v.push_back(transformation_v[0]);
                        t_i.theta_Y_v.push_back(transformation_v[1]);
                        t_i.theta_Z_v.push_back(transformation_v[2]);
                        t_i.t_X_v.push_back(transformation_v[3]);
                        t_i.t_Y_v.push_back(transformation_v[4]);
                        t_i.t_Z_v.push_back(transformation_v[5]);
                        t_i.subsets_and_seed.push_back(ss_sr);
                        t_i.angles_v.push_back(angle);
                        t_i.psi_v.push_back(psi);
                        t_i.votes = v_stp;
                        theta_Y_it->second[bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                        // Add the bin to the aux accumulator
                        std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                        aux_binning.push_back(binning);
                    }
                } else {    // Angle theta_Y was not found. Thus, it is a new transformation
                    ea_transformation_info t_i = ea_transformation_info();
                    // Initialize all containers as empty and zero votes
                    t_i.theta_X_v.clear();
                    t_i.theta_Y_v.clear();
                    t_i.theta_Z_v.clear();
                    t_i.t_X_v.clear();
                    t_i.t_Y_v.clear();
                    t_i.t_Z_v.clear();
                    t_i.subsets_and_seed.clear();
                    t_i.angles_v.clear();
                    t_i.psi_v.clear();
                    t_i.votes = 0;
                    // Add the information of the very first transformation and give one vote
                    std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                    t_i.theta_X_v.push_back(transformation_v[0]);
                    t_i.theta_Y_v.push_back(transformation_v[1]);
                    t_i.theta_Z_v.push_back(transformation_v[2]);
                    t_i.t_X_v.push_back(transformation_v[3]);
                    t_i.t_Y_v.push_back(transformation_v[4]);
                    t_i.t_Z_v.push_back(transformation_v[5]);
                    t_i.subsets_and_seed.push_back(ss_sr);
                    t_i.angles_v.push_back(angle);
                    t_i.psi_v.push_back(psi);
                    t_i.votes = v_stp;
                    theta_Z_it->second[bin_theta_Y][bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                    // Add the bin to the aux accumulator
                    std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                    aux_binning.push_back(binning);
                }
            } else {    // Angle theta_Z was not found. Thus, it is a new transformation
                ea_transformation_info t_i = ea_transformation_info();
                // Initialize all containers as empty and zero votes
                t_i.theta_X_v.clear();
                t_i.theta_Y_v.clear();
                t_i.theta_Z_v.clear();
                t_i.t_X_v.clear();
                t_i.t_Y_v.clear();
                t_i.t_Z_v.clear();
                t_i.subsets_and_seed.clear();
                t_i.angles_v.clear();
                t_i.psi_v.clear();
                t_i.votes = 0;
                // Add the information of the very first transformation and give one vote
                std::vector<int> ss_sr {subsets_i[0], subsets_i[1], seed_radius};
                t_i.theta_X_v.push_back(transformation_v[0]);
                t_i.theta_Y_v.push_back(transformation_v[1]);
                t_i.theta_Z_v.push_back(transformation_v[2]);
                t_i.t_X_v.push_back(transformation_v[3]);
                t_i.t_Y_v.push_back(transformation_v[4]);
                t_i.t_Z_v.push_back(transformation_v[5]);
                t_i.subsets_and_seed.push_back(ss_sr);
                t_i.angles_v.push_back(angle);
                t_i.psi_v.push_back(psi);
                t_i.votes = v_stp;
                accumulator[bin_theta_Z][bin_theta_Y][bin_theta_X][bin_t_X][bin_t_Y][bin_t_Z] = t_i;
                // Add the bin to the aux accumulator
                std::vector<int> binning {bin_theta_Z, bin_theta_Y, bin_theta_X, bin_t_X, bin_t_Y, bin_t_Z};
                aux_binning.push_back(binning);
            }
            break;
        }
        default:
            assert(false);
    }
}


float GetFrequentAngle(std::vector<float>& angles_v) {
    // Insert all elements in hash
    std::unordered_map<float, int> hash;
    for (auto i : angles_v)
        hash[i]++;
    // Find the max frequency
    int max_count = 0;
    float res = -1.0;
    for (auto i : hash) {
        if (max_count < i.second) {
            res = i.first;
            max_count = i.second;
        }
    }
    return res;
}


void VotesCounting(accumulator_7D &accumulator, w_vector_int &aux_binning, std::vector<int> &votes_v, 
                   std::vector<double> &psi_v, std::vector<float> &angles_v) {
    votes_v.resize(aux_binning.size());
    psi_v.resize(aux_binning.size());
    angles_v.resize(aux_binning.size());
    for (unsigned int i = 0; i < aux_binning.size(); ++i) {
        auto q_W_it = accumulator.find(aux_binning[i][0]);
        if (q_W_it != accumulator.end()) {
            auto q_X_it = q_W_it->second.find(aux_binning[i][1]);
            if (q_X_it != q_W_it->second.end()) {
                auto q_Y_it = q_X_it->second.find(aux_binning[i][2]);
                if (q_Y_it != q_X_it->second.end()) {
                    auto q_Z_it = q_Y_it->second.find(aux_binning[i][3]);
                    if (q_Z_it != q_Y_it->second.end()) {
                        auto t_X_it = q_Z_it->second.find(aux_binning[i][4]);
                        if (t_X_it != q_Z_it->second.end()) {
                            auto t_Y_it = t_X_it->second.find(aux_binning[i][5]);
                            if (t_Y_it != t_X_it->second.end()) {
                                auto t_Z_it = t_Y_it->second.find(aux_binning[i][6]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    votes_v[i] = t_Z_it->second.votes;
                                    psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                           t_Z_it->second.psi_v.end(), 0.0)) /
                                                                   (double)t_Z_it->second.psi_v.size();
                                    angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][6]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                     << "] for 'tY' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                 << "] for 'tX' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                             << "] for 'qZ' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                         << "] for 'qY' does not exist in the std::map" << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                     << "] for 'qX' does not exist in the std::map" << std::endl;
        } else
            std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                 << "] for 'qW' does not exist in the std::map" << std::endl;
    }
}


void VotesCounting(accumulator_6D &accumulator, w_vector_int &aux_binning, int order, std::vector<int> &votes_v, 
                   std::vector<double> &psi_v, std::vector<float> &angles_v) {
    psi_v.resize(aux_binning.size());
    votes_v.resize(aux_binning.size());
    angles_v.resize(aux_binning.size());
    switch (order) {
        case 1: {   // XYZ order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_X
                auto theta_X_it = accumulator.find(aux_binning[i][0]);
                if (theta_X_it != accumulator.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_X_it->second.find(aux_binning[i][1]);
                    if (theta_Y_it != theta_X_it->second.end()) {
                        // Look for theta_Z
                        auto theta_Z_it = theta_Y_it->second.find(aux_binning[i][2]);
                        if (theta_Z_it != theta_Y_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_Z_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_Z_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_Z' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_Y' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_X' does not exist in the std::map" << std::endl;
            }
            break;
        }
        case 2: {   // XZY order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_X
                auto theta_X_it = accumulator.find(aux_binning[i][0]);
                if (theta_X_it != accumulator.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_X_it->second.find(aux_binning[i][1]);
                    if (theta_Z_it != theta_X_it->second.end()) {
                        // Look for theta_Y
                        auto theta_Y_it = theta_Z_it->second.find(aux_binning[i][2]);
                        if (theta_Y_it != theta_Z_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_Y_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_Y_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_Y' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_Z' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_X' does not exist in the std::map" << std::endl;
            }
            break;
        }
        case 3: {   // YXZ order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_Y
                auto theta_Y_it = accumulator.find(aux_binning[i][0]);
                if (theta_Y_it != accumulator.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Y_it->second.find(aux_binning[i][1]);
                    if (theta_X_it != theta_Y_it->second.end()) {
                        // Look for theta_Z
                        auto theta_Z_it = theta_X_it->second.find(aux_binning[i][2]);
                        if (theta_Z_it != theta_X_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_Z_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_Z_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_Z' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_X' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_Y' does not exist in the std::map" << std::endl;
            }
            break;
        }
        case 4: {   // YZX order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_Y
                auto theta_Y_it = accumulator.find(aux_binning[i][0]);
                if (theta_Y_it != accumulator.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_Y_it->second.find(aux_binning[i][1]);
                    if (theta_Z_it != theta_Y_it->second.end()) {
                        // Look for theta_X
                        auto theta_X_it = theta_Z_it->second.find(aux_binning[i][2]);
                        if (theta_X_it != theta_Z_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_X_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_X_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_X' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_Z' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_Y' does not exist in the std::map" << std::endl;
            }
            break;
        }
        case 5: {   // ZXY order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_Z
                auto theta_Z_it = accumulator.find(aux_binning[i][0]);
                if (theta_Z_it != accumulator.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Z_it->second.find(aux_binning[i][1]);
                    if (theta_X_it != theta_Z_it->second.end()) {
                        // Look for theta_Y
                        auto theta_Y_it = theta_X_it->second.find(aux_binning[i][2]);
                        if (theta_Y_it != theta_X_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_Y_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_Y_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_Y' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_X' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_Z' does not exist in the std::map" << std::endl;
            }
            break;
        }
        case 6: {   // ZYX order
            for (unsigned int i = 0; i < aux_binning.size(); ++i) {
                // Look for theta_Z
                auto theta_Z_it = accumulator.find(aux_binning[i][0]);
                if (theta_Z_it != accumulator.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_Z_it->second.find(aux_binning[i][1]);
                    if (theta_Y_it != theta_Z_it->second.end()) {
                        // Look for theta_X
                        auto theta_X_it = theta_Y_it->second.find(aux_binning[i][2]);
                        if (theta_X_it != theta_Y_it->second.end()) {
                            // Look for t_X
                            auto t_X_it = theta_X_it->second.find(aux_binning[i][3]);
                            if (t_X_it != theta_X_it->second.end()) {
                                // Look for t_Y
                                auto t_Y_it = t_X_it->second.find(aux_binning[i][4]);
                                if (t_Y_it != t_X_it->second.end()) {
                                    // Look for t_Z
                                    auto t_Z_it = t_Y_it->second.find(aux_binning[i][5]);
                                    if (t_Z_it != t_Y_it->second.end()) {
                                        votes_v[i] = t_Z_it->second.votes;
                                        psi_v[i] = (accumulate(t_Z_it->second.psi_v.begin(),
                                                               t_Z_it->second.psi_v.end(), 0.0)) /
                                                                       (double)t_Z_it->second.psi_v.size();
                                        angles_v[i] = GetFrequentAngle(t_Z_it->second.angles_v);
                                    } else
                                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][5]
                                             << "] for 'tZ' does not exist in the std::map." << std::endl;
                                } else
                                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][4]
                                         << "] for 'tY' does not exist in the std::map" << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << aux_binning[i][3]
                                     << "] for 'tX' does not exist in the std::map" << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << aux_binning[i][2]
                                 << "] for 'theta_X' does not exist in the std::map" << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << aux_binning[i][1]
                             << "] for 'theta_Y' does not exist in the std::map" << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << aux_binning[i][0]
                         << "] for 'theta_X' does not exist in the std::map" << std::endl;
            }
            break;
        }
        default:
            assert(false);
    }
}


void PrintAccumulatorAndMetrics(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v,
                                std::vector<float> &angles_v, std::stringstream &file_name) {
    std::ofstream output_file(file_name.str(), std::ios_base::app | std::ios_base::out);
    for (unsigned int i = 0; i < aux_binning.size(); ++i) {
        output_file << i + 1 << ",";
        for (int j : aux_binning[i])
            output_file << j << ",";
        output_file << votes_v[i] << "," << angles_v[i] << "," << psi_v[i] << std::endl;
    }
}


void FindBestBinByVotes(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v,
                        std::vector<int> &best_bin, std::ofstream& summary_file) {
    int max_votes = -std::numeric_limits<int>::max();
    double min_psi = std::numeric_limits<double>::max();
    int max_id_1 = std::numeric_limits<int>::max();
    for (unsigned int i = 0; i < votes_v.size(); ++i) {
        if (votes_v[i] > max_votes) {
            min_psi = psi_v[i];
            max_votes = votes_v[i];
            max_id_1 = i;
        } else if (votes_v[i] == max_votes) {
            if (psi_v[i] <= min_psi) {
                min_psi = psi_v[i];
                max_votes = votes_v[i];
                max_id_1 = i;
            }
        }
    }
    best_bin = aux_binning[max_id_1];
    std::cout << "\nBest bin by votes is No. " << max_id_1 + 1 << ":\n\t";
    summary_file << "\nBest bin by votes is No. " << max_id_1 + 1 << ":\n\t";
    for (int j : best_bin) {
        std::cout << "[" << j << "]";
        summary_file << "[" << j << "]";
    }
    std::cout << " with " << max_votes << " votes." << std::endl;
    summary_file << " with " << max_votes << " votes." << std::endl;
}


void FindBestBinByVotes(w_vector_int &aux_binning, std::vector<int> &votes_v, std::vector<double> &psi_v,
                        std::vector<int> &best_bin, int &bin_id, int &votes) {
    int max_votes = -std::numeric_limits<int>::max();
    double min_psi = std::numeric_limits<double>::max();
    int max_id = std::numeric_limits<int>::max();
    for (unsigned int i = 0; i < votes_v.size(); ++i) {
        if (votes_v[i] > max_votes) {
            min_psi = psi_v[i];
            max_votes = votes_v[i];
            max_id = i;
        } else if (votes_v[i] == max_votes) {
            if (psi_v[i] <= min_psi) {
                min_psi = psi_v[i];
                max_votes = votes_v[i];
                max_id = i;
            }
        }
    }
    best_bin = aux_binning[max_id];
    bin_id = max_id + 1;
    votes = max_votes;
}

void FindBestBinByPSI(w_vector_int &aux_binning, std::vector<double> &psi_v, std::vector<int> &best_bin,
                      std::ofstream& summary_file) {
    double min_psi = std::numeric_limits<double>::max();
    int min_id = std::numeric_limits<int>::max();
    for (unsigned int i = 0; i < psi_v.size(); ++i) {
        if (psi_v[i] < min_psi) {
            min_psi = psi_v[i];
            min_id = i;
        }
    }
    best_bin = aux_binning[min_id];
    std::cout << "\nBest bin by PSI is No. " << min_id + 1 << ":\n\t";
    summary_file << "\nBest bin by PSI is No. " << min_id + 1 << ":\n\t";
    for (int j : best_bin) {
        std::cout << "[" << j << "]";
        summary_file << "[" << j << "]";
    }
    std::cout << " with a PSI = " << min_psi << std::endl;
    summary_file << " with a PSI = " << min_psi << std::endl;
}


void GetBestBinTransformation(accumulator_7D &accumulator, std::vector<int> &best_bin, std::vector<std::vector<double>> &best_transformations_v,
                              std::vector<std::vector<int>> &best_corresponding_subsets_v, std::map<int, std::vector<int>> &seed_subsets_map) {
    std::vector<double> q_W_v, q_X_v, q_Y_v, q_Z_v, t_X_v, t_Y_v, t_Z_v;
    auto q_W_it = accumulator.find(best_bin[0]);
    if (q_W_it != accumulator.end()) {
        auto q_X_it = q_W_it->second.find(best_bin[1]);
        if (q_X_it != q_W_it->second.end()) {
            auto q_Y_it = q_X_it->second.find(best_bin[2]);
            if (q_Y_it != q_X_it->second.end()) {
                auto q_Z_it = q_Y_it->second.find(best_bin[3]);
                if (q_Z_it != q_Y_it->second.end()) {
                    auto t_X_it = q_Z_it->second.find(best_bin[4]);
                    if (t_X_it != q_Z_it->second.end()) {
                        auto t_Y_it = t_X_it->second.find(best_bin[5]);
                        if (t_Y_it != t_X_it->second.end()) {
                            auto t_Z_it = t_Y_it->second.find(best_bin[6]);
                            if (t_Z_it != t_Y_it->second.end()) {
                                // Copying the information from the accumulator
                                q_W_v = t_Z_it->second.q_W_v;
                                q_X_v = t_Z_it->second.q_X_v;
                                q_Y_v = t_Z_it->second.q_Y_v;
                                q_Z_v = t_Z_it->second.q_Z_v;
                                t_X_v = t_Z_it->second.t_X_v;
                                t_Y_v = t_Z_it->second.t_Y_v;
                                t_Z_v = t_Z_it->second.t_Z_v;
                                best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                // Generating the ordered seed-subsets std::map
                                int seed_id, subset_id;
                                for (auto & i : best_corresponding_subsets_v) {
                                    seed_id = i[2];
                                    subset_id = i[0];
                                    auto s_ss_it = seed_subsets_map.find(seed_id);
                                    if (s_ss_it != seed_subsets_map.end())
                                        seed_subsets_map[seed_id].push_back(subset_id);
                                    else {
                                        std::vector<int> subsets_ids {subset_id};
                                        seed_subsets_map[seed_id] = subsets_ids;
                                    }
                                }
                                // Re-converting the transformations to matrices
                                best_transformations_v.resize(q_W_v.size());
                                for (unsigned int i = 0; i < q_W_v.size(); ++i) {
                                    std::vector<double> c_transformation(7);
                                    c_transformation[0] = q_W_v[i];
                                    c_transformation[1] = q_X_v[i];
                                    c_transformation[2] = q_Y_v[i];
                                    c_transformation[3] = q_Z_v[i];
                                    c_transformation[4] = t_X_v[i];
                                    c_transformation[5] = t_Y_v[i];
                                    c_transformation[6] = t_Z_v[i];
                                    best_transformations_v[i] = c_transformation;
                                }
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[6]
                                     << "] for 'tZ' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                 << "] for 'tY' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[4]
                             << "] for 'tX' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[3]
                         << "] for 'qZ' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[2]
                     << "] for 'qY' does not exist in the std::map." << std::endl;
        } else
            std::cout << "ERROR:\n\tBin [" << best_bin[1]
                 << "] for 'qX' does not exist in the std::map." << std::endl;
    } else
        std::cout << "ERROR:\n\tBin [" << best_bin[0]
             << "] for 'qW' does not exist in the std::map." << std::endl;
}


void GetBestBinTransformation(std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, std::map<int, ea_transformation_info>>>>>> &accumulator,
                              int order, std::vector<int> &best_bin, std::vector<std::vector<double>> &best_transformations_v,
                                 std::vector<std::vector<int>> &best_corresponding_subsets_v,
                                 std::map<int, std::vector<int>> &seed_subsets_map) {
    std::vector<double> theta_X_v, theta_Y_v, theta_Z_v, t_X_v, t_Y_v, t_Z_v;
    switch (order) {
        case 1: {   // XYZ order
            // Look for theta_X
            auto theta_X_it = accumulator.find(best_bin[0]);
            if (theta_X_it != accumulator.end()) {
                // Look for theta_Y
                auto theta_Y_it = theta_X_it->second.find(best_bin[1]);
                if (theta_Y_it != theta_X_it->second.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_Y_it->second.find(best_bin[2]);
                    if (theta_Z_it != theta_Y_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_Z_it->second.find(best_bin[3]);
                        if (t_X_it != theta_Z_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_Z' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_Y' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_X' does not exist in the std::map." << std::endl;
            break;
        }
        case 2: {   // XZY order
            // Look for theta_X
            auto theta_X_it = accumulator.find(best_bin[0]);
            if (theta_X_it != accumulator.end()) {
                // Look for theta_Z
                auto theta_Z_it = theta_X_it->second.find(best_bin[1]);
                if (theta_Z_it != theta_X_it->second.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_Z_it->second.find(best_bin[2]);
                    if (theta_Y_it != theta_Z_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_Y_it->second.find(best_bin[3]);
                        if (t_X_it != theta_Y_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_Y' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_Z' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_X' does not exist in the std::map." << std::endl;
            break;
        }
        case 3: {   // YXZ order
            // Look for theta_Y
            auto theta_Y_it = accumulator.find(best_bin[0]);
            if (theta_Y_it != accumulator.end()) {
                // Look for theta_X
                auto theta_X_it = theta_Y_it->second.find(best_bin[1]);
                if (theta_X_it != theta_Y_it->second.end()) {
                    // Look for theta_Z
                    auto theta_Z_it = theta_X_it->second.find(best_bin[2]);
                    if (theta_Z_it != theta_X_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_Z_it->second.find(best_bin[3]);
                        if (t_X_it != theta_Z_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_Z' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_X' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_Y' does not exist in the std::map." << std::endl;
            break;
        }
        case 4: {   // YZX order
            // Look for theta_Y
            auto theta_Y_it = accumulator.find(best_bin[0]);
            if (theta_Y_it != accumulator.end()) {
                // Look for theta_Z
                auto theta_Z_it = theta_Y_it->second.find(best_bin[1]);
                if (theta_Z_it != theta_Y_it->second.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Z_it->second.find(best_bin[2]);
                    if (theta_X_it != theta_Z_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_X_it->second.find(best_bin[3]);
                        if (t_X_it != theta_X_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_X' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_Z' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_Y' does not exist in the std::map." << std::endl;
            break;
        }
        case 5: {   // ZXY order
            // Look for theta_Z
            auto theta_Z_it = accumulator.find(best_bin[0]);
            if (theta_Z_it != accumulator.end()) {
                // Look for theta_X
                auto theta_X_it = theta_Z_it->second.find(best_bin[1]);
                if (theta_X_it != theta_Z_it->second.end()) {
                    // Look for theta_Y
                    auto theta_Y_it = theta_X_it->second.find(best_bin[2]);
                    if (theta_Y_it != theta_X_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_Y_it->second.find(best_bin[3]);
                        if (t_X_it != theta_Y_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_Y' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_X' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_Z' does not exist in the std::map." << std::endl;
            break;
        }
        case 6: {   // ZYX order
            // Look for theta_Z
            auto theta_Z_it = accumulator.find(best_bin[0]);
            if (theta_Z_it != accumulator.end()) {
                // Look for theta_Y
                auto theta_Y_it = theta_Z_it->second.find(best_bin[1]);
                if (theta_Y_it != theta_Z_it->second.end()) {
                    // Look for theta_X
                    auto theta_X_it = theta_Y_it->second.find(best_bin[2]);
                    if (theta_X_it != theta_Y_it->second.end()) {
                        // Look for t_X
                        auto t_X_it = theta_X_it->second.find(best_bin[3]);
                        if (t_X_it != theta_X_it->second.end()) {
                            // Look for t_Y
                            auto t_Y_it = t_X_it->second.find(best_bin[4]);
                            if (t_Y_it != t_X_it->second.end()) {
                                // Look for t_Z
                                auto t_Z_it = t_Y_it->second.find(best_bin[5]);
                                if (t_Z_it != t_Y_it->second.end()) {
                                    // Copying the information from the accumulator
                                    theta_X_v = t_Z_it->second.theta_X_v;
                                    theta_Y_v = t_Z_it->second.theta_Y_v;
                                    theta_Z_v = t_Z_it->second.theta_Z_v;
                                    t_X_v = t_Z_it->second.t_X_v;
                                    t_Y_v = t_Z_it->second.t_Y_v;
                                    t_Z_v = t_Z_it->second.t_Z_v;
                                    best_corresponding_subsets_v = t_Z_it->second.subsets_and_seed;
                                    // Generating the ordered seed-subsets std::map
                                    int seed_id, subset_id;
                                    for (auto & i : best_corresponding_subsets_v) {
                                        seed_id = i[2];
                                        subset_id = i[0];
                                        auto s_ss_it = seed_subsets_map.find(seed_id);
                                        if (s_ss_it != seed_subsets_map.end())
                                            seed_subsets_map[seed_id].push_back(subset_id);
                                        else {
                                            std::vector<int> subsets_ids {subset_id};
                                            seed_subsets_map[seed_id] = subsets_ids;
                                        }
                                    }
                                    // Re-converting the transformations to matrices
                                    best_transformations_v.resize(theta_X_v.size());
                                    for (unsigned int i = 0; i < theta_X_v.size(); ++i) {
                                        std::vector<double> c_transformation(6);
                                        c_transformation[0] = theta_X_v[i];
                                        c_transformation[1] = theta_Y_v[i];
                                        c_transformation[2] = theta_Z_v[i];
                                        c_transformation[3] = t_X_v[i];
                                        c_transformation[4] = t_Y_v[i];
                                        c_transformation[5] = t_Z_v[i];
                                        best_transformations_v[i] = c_transformation;
                                    }
                                } else
                                    std::cout << "ERROR:\n\tBin [" << best_bin[5]
                                         << "] for 'tZ' does not exist in the std::map." << std::endl;
                            } else
                                std::cout << "ERROR:\n\tBin [" << best_bin[4]
                                     << "] for 'tY' does not exist in the std::map." << std::endl;
                        } else
                            std::cout << "ERROR:\n\tBin [" << best_bin[3]
                                 << "] for 'tX' does not exist in the std::map." << std::endl;
                    } else
                        std::cout << "ERROR:\n\tBin [" << best_bin[2]
                             << "] for 'theta_X' does not exist in the std::map." << std::endl;
                } else
                    std::cout << "ERROR:\n\tBin [" << best_bin[1]
                         << "] for 'theta_Y' does not exist in the std::map." << std::endl;
            } else
                std::cout << "ERROR:\n\tBin [" << best_bin[0]
                     << "] for 'theta_Z' does not exist in the std::map." << std::endl;
            break;
        }
        default:
            assert(false);
    }
}


void ComputeAverageTransformation(std::vector<std::vector<double>> &transformations_v, std::vector<double> &av_transformation) {
    double q_W_sum = 0.0, q_X_sum = 0.0, q_Y_sum = 0.0, q_Z_sum = 0.0, t_X_sum = 0.0, t_Y_sum = 0.0, t_Z_sum = 0.0;
    for (auto & i : transformations_v) {
        q_W_sum += i[0];
        q_X_sum += i[1];
        q_Y_sum += i[2];
        q_Z_sum += i[3];
        t_X_sum += i[4];
        t_Y_sum += i[5];
        t_Z_sum += i[6];
    }
    av_transformation.resize(7);
    av_transformation[0] = q_W_sum / (double)transformations_v.size();
    av_transformation[1] = q_X_sum / (double)transformations_v.size();
    av_transformation[2] = q_Y_sum / (double)transformations_v.size();
    av_transformation[3] = q_Z_sum / (double)transformations_v.size();
    av_transformation[5] = t_Y_sum / (double)transformations_v.size();
    av_transformation[4] = t_X_sum / (double)transformations_v.size();
    av_transformation[6] = t_Z_sum / (double)transformations_v.size();
}

void ComputeAverageTransformationEA(std::vector<std::vector<double>> &transformations_v, std::vector<double> &av_transformation) {
    double theta_X_sum = 0.0, theta_Y_sum = 0.0, theta_Z_sum = 0.0, t_X_sum = 0.0, t_Y_sum = 0.0, t_Z_sum = 0.0;
    for (auto & i : transformations_v) {
        theta_X_sum += i[0];
        theta_Y_sum += i[1];
        theta_Z_sum += i[2];
        t_X_sum += i[3];
        t_Y_sum += i[4];
        t_Z_sum += i[5];
    }
    av_transformation.resize(6);
    av_transformation[0] = theta_X_sum / (double)transformations_v.size();
    av_transformation[1] = theta_Y_sum / (double)transformations_v.size();
    av_transformation[2] = theta_Z_sum / (double)transformations_v.size();
    av_transformation[3] = t_X_sum / (double)transformations_v.size();
    av_transformation[4] = t_Y_sum / (double)transformations_v.size();
    av_transformation[5] = t_Z_sum / (double)transformations_v.size();
}


void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* nothing) {}


void DynamicVisualizer(std::vector<std::vector<double>> &transformations_v_a, std::vector<double> &transformation_v_b,
                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud,
                       std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &corresponding_subsets_v) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr_b (new pcl::PointCloud<pcl::PointXYZ> ());
    Eigen::Matrix4d transformation_matrix_b;
    // Best by PSI
    VectorToMatrix(transformation_v_b, transformation_matrix_b);
    transformPointCloud(*source_cloud, *source_cloud_tr_b, transformation_matrix_b);
    std::stringstream votes_text, psi_text;
    votes_text << "Transformations by VOTES";
    psi_text << "Best transformation by PSI";
    int v_1(0), v_2(0);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("leading_alignments"));
    vs->createViewPort(0.0, 0.0, 0.5, 1.0, v_1);
    vs->createViewPort(0.5, 0.0, 1.0, 1.0, v_2);
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    // Text
    vs->addText(votes_text.str(), 10, 30, 15, 0.0, 0.0, 0.0, "votes_result", v_1);
    vs->addText(psi_text.str(), 10, 30, 15, 0.0, 0.0, 0.0, "psi_result", v_2);
    // Visualizer 1
    vs->addPointCloud<pcl::PointXYZ>(source_cloud, "source_v_1", v_1);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_v_1");
    vs->addPointCloud<pcl::PointXYZ>(target_cloud, "target_v_1", v_1);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target_v_1");
    vs->addPointCloud<pcl::PointXYZ>(source_cloud, "subset_v_1", v_1);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "subset_v_1");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "subset_v_1");
    // Visualizer 2
    vs->addPointCloud<pcl::PointXYZ>(source_cloud_tr_b, "source_v_2", v_2);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_v_2");
    vs->addPointCloud<pcl::PointXYZ>(target_cloud, "target_v_2", v_2);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target_v_2");
    vs->resetCamera();
    vs->registerKeyboardCallback(&KeyboardEventOccurred, (void*) nullptr);
    // Best by votes
    Eigen::Matrix4d transformation_matrix_a;
    for (unsigned int i = 0; i < transformations_v_a.size(); ++i) {
        votes_text.str("");
        votes_text << "Transformations by VOTES (" << i << ")";
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr_a (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointXYZ>::Ptr subset_tr (new pcl::PointCloud<pcl::PointXYZ> ());
        transformation_matrix_a = Eigen::Matrix4d::Identity();
        VectorToMatrix(transformations_v_a[i], transformation_matrix_a);
        transformPointCloud(*source_cloud, *source_cloud_tr_a, transformation_matrix_a);
        transformPointCloud(*corresponding_subsets_v[i], *subset_tr, transformation_matrix_a);
        vs->updatePointCloud(source_cloud_tr_a, "source_v_1");
        vs->updatePointCloud(subset_tr, "subset_v_1");
        vs->updateText(votes_text.str(), 10, 30, 15, 0.0, 0.0, 0.0, "votes_result");
        vs->spinOnce(10);
    }
    vs->close();
}

void StaticVisualizer(Eigen::Matrix4d &tr_matrix, const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud, float inlier_threshold) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr_a (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr overlap_cloud_a (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    // Averaged best
    pcl::transformPointCloud(*source_cloud, *source_cloud_tr_a, tr_matrix);
    for (auto point : source_cloud_tr_a->points) {
        std::vector<int> nearest_neighbor (1);
        std::vector<float> nearest_neighbor_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nearest_neighbor, nearest_neighbor_sqr_distance);
        if (!nearest_neighbor.empty()) {
            overlap_cloud_a->points.push_back(point);
        }
    }
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("final_alignments"));
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    vs->addPointCloud<pcl::PointXYZ>(source_cloud_tr_a, "source");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source");
    vs->addPointCloud<pcl::PointXYZ>(target_cloud, "target");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target");
    vs->addPointCloud<pcl::PointXYZ>(overlap_cloud_a, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "overlap");
    vs->resetCamera();
    vs->registerKeyboardCallback(&KeyboardEventOccurred, (void*) nullptr);
    while (!vs->wasStopped()) {
        vs->spinOnce(100);
        if (close_visualizer) {
            close_visualizer = false;
            vs->close();
        }
    }
}


void StaticVisualizer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud, 
                      float inlier_threshold) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr overlap_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    for (auto point : source_cloud->points) {
        std::vector<int> nearest_neighbor (1);
        std::vector<float> nearest_neighbor_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nearest_neighbor, nearest_neighbor_sqr_distance);
        if (!nearest_neighbor.empty()) {
            overlap_cloud->points.push_back(point);
        }
    }
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("final_alignments"));
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    vs->addPointCloud<pcl::PointXYZ>(source_cloud, "source");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source");
    vs->addPointCloud<pcl::PointXYZ>(target_cloud, "target");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target");
    vs->addPointCloud<pcl::PointXYZ>(overlap_cloud, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "overlap");
    vs->resetCamera();
    vs->registerKeyboardCallback(&KeyboardEventOccurred, (void*) nullptr);
    while (!vs->wasStopped()) {
        vs->spinOnce(100);
        if (close_visualizer) {
            close_visualizer = false;
            vs->close();
        }
    }
}

void StaticVisualizerPOV(Eigen::Matrix4d &transformation_a, const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud, float inlier_threshold,
                         std::stringstream& file_base_name) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_tr_a (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr overlap_cloud_a (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(target_cloud);
    // Averaged best
    pcl::transformPointCloud(*source_cloud, *source_cloud_tr_a, transformation_a);
    for (auto point : source_cloud_tr_a->points) {
        std::vector<int> nearest_neighbor (1);
        std::vector<float> nearest_neighbor_sqr_distance (1);
        tree.radiusSearch(point, inlier_threshold, nearest_neighbor, nearest_neighbor_sqr_distance);
        if (!nearest_neighbor.empty()) {
            overlap_cloud_a->points.push_back(point);
        }
    }
    std::stringstream pos_name, scn_shot_name;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("final_alignments"));
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    vs->addCoordinateSystem(0.01f, "reference");
    vs->initCameraParameters();
    vs->setCameraFieldOfView(0.20);
    vs->setCameraClipDistances(0.00, 50.0);
    vs->addPointCloud<pcl::PointXYZ>(source_cloud_tr_a, "source");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source");
    vs->addPointCloud<pcl::PointXYZ>(target_cloud, "target");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target");
    vs->addPointCloud<pcl::PointXYZ>(overlap_cloud_a, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "overlap");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "overlap");
    std::vector<std::vector<float>> cam_positions;
    cam_positions.resize(3);
    cam_positions[0] = {0.0, 0.0, 0.0};
    cam_positions[1] = {180.0, 0.0, 0.0};
    cam_positions[2] = {0.0, 180.0, 0.0};
    for (unsigned int p_i = 0; p_i < cam_positions.size(); ++p_i) {
        pos_name.str("");
        scn_shot_name.str("");
        switch (p_i) {
            case 0 :
                pos_name << "FRONT_VIEW";
                break;
            case 1 :
                pos_name << "SIDE_VIEW";
                break;
            case 2 :
                pos_name << "TOP_VIEW";
                break;
            default:
                assert(false);
        }
        scn_shot_name << file_base_name.str() << "_RESULTS_" << pos_name.str() << ".png";
        vs->setCameraPosition(0.0, 0.0, 30.0,
                              cam_positions[p_i][0] + 1, cam_positions[p_i][1] + 1, cam_positions[p_i][2],
                              0.0, 1.0, 0.0);
        vs->resetCamera();
        vs->spinOnce(100);
        vs->saveScreenshot(scn_shot_name.str());
    }
    vs->close();
}


void Voxelization(size_t leaf_size, float resolution, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointXYZ &min_point,
                  const pcl::PointXYZ &max_point, std::vector<std::vector<int>> &indices_v,
                  std::vector<std::vector<Eigen::Vector3f>> &voxels_min_max_pt) {
    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.defineBoundingBox(min_point.x, min_point.y, min_point.z, max_point.x, max_point.y, max_point.z);
    octree.enableDynamicDepth(leaf_size);
    octree.addPointsFromInputCloud();
    std::cout << "\t" <<octree.getLeafCount() << " voxels defined." << std::endl;
    int leaf_count = 1;
    for (auto it = octree.leaf_depth_begin(), it_end = octree.leaf_depth_end(); it != it_end; ++it) {
        std::vector<int> idx_v;
        Eigen::Vector3f v_min_pt, v_max_pt;
        pcl::octree::OctreeContainerPointIndices& container = it.getLeafContainer();
        octree.getVoxelBounds(it, v_min_pt, v_max_pt);
        container.getPointIndices(idx_v);
        std::vector<Eigen::Vector3f> min_max_pt {v_min_pt, v_max_pt};
        voxels_min_max_pt.push_back(min_max_pt);
        indices_v.push_back(idx_v);
        leaf_count++;
    }
}


void HeatMapVisualizer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<std::vector<int>> &subsets_points_idx,
                       std::stringstream &file_base_name, std::ofstream &summary_file) {
    std::map<int, int> point_heat_map;
    // Set each point heat value as 0
    for (unsigned int p_id = 0; p_id < cloud->points.size(); ++p_id)
        point_heat_map[p_id] = 0;
    // Set heat value
    for (auto & ss_i : subsets_points_idx) {
        for (int p_i : ss_i) {
            auto it = point_heat_map.find(p_i);
            if (it != point_heat_map.end())
                point_heat_map[p_i] ++;
        }
    }
    // Get the max heat and print points data
    std::stringstream point_data_file;
    point_data_file << file_base_name.str() << "_POINTS_HEAT.csv";
    std::ofstream point_data_file_header(point_data_file.str());
    point_data_file_header << "point, heat" << std::endl;
    std::ofstream output_f(point_data_file.str(), std::ios_base::app | std::ios_base::out);
    int max_votes = -std::numeric_limits<int>::max();
    for (auto & p_id : point_heat_map) {
        if (p_id.second > max_votes)
            max_votes = p_id.second;
        output_f << p_id.first << "," << p_id.second << std::endl;
    }
    // Set the heat ranges
    float heat_step = (float)max_votes / 6;
    float blue_max = 0 + heat_step;
    float cyan_max = blue_max + heat_step;
    float green_max = cyan_max + heat_step;
    float yellow_max = green_max + heat_step;
    float orange_max = yellow_max + heat_step;
    float red_max = orange_max + heat_step;
    std::cout << "\nHeat scale:"
            "\n\tRed\t\t...\t(" << orange_max << ", " << red_max << "]"
            "\n\tOrange\t...\t(" << yellow_max << ", " << orange_max << "]"
            "\n\tYellow\t...\t(" << green_max << ", " << yellow_max << "]"
            "\n\tGreen\t...\t(" << cyan_max << ", " << green_max << "]"
            "\n\tCyan\t...\t(" << blue_max << ", " << cyan_max << "]"
            "\n\tBlue\t...\t[" << 0 << ", " << blue_max << "]" << std::endl;
    summary_file << "\nHeat scale:"
                    "\n\tRed\t\t...\t(" << orange_max << ", " << red_max << "]"
                    "\n\tOrange\t...\t(" << yellow_max << ", " << orange_max << "]"
                    "\n\tYellow\t...\t(" << green_max << ", " << yellow_max << "]"
                    "\n\tGreen\t...\t(" << cyan_max << ", " << green_max << "]"
                    "\n\tCyan\t...\t(" << blue_max << ", " << cyan_max << "]"
                    "\n\tBlue\t...\t[" << 0 << ", " << blue_max << "]" << std::endl;
    // Build the point clouds according to their heat values
    pcl::PointCloud<pcl::PointXYZ>::Ptr blue_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cyan_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr green_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr yellow_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr orange_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr red_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr overlapping_areas (new pcl::PointCloud<pcl::PointXYZ> ());
    for (auto & p_id : point_heat_map) {
        if ((float)p_id.second <= blue_max)
            blue_points->points.push_back(cloud->points[p_id.first]);
        else if (((float)p_id.second > blue_max) && ((float)p_id.second <= cyan_max)) {
            cyan_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > cyan_max) && ((float)p_id.second <= green_max)) {
            green_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > green_max) && ((float)p_id.second <= yellow_max)) {
            yellow_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > yellow_max) && ((float)p_id.second <= orange_max)) {
            orange_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > orange_max) && ((float)p_id.second <= red_max)) {
            red_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        }
    }
    std::cout << "\nPoints assortment:"
            "\n\tRed points:\t" << red_points->points.size()
         << "\n\tOrange points:\t" << orange_points->points.size()
         << "\n\tYellow points:\t" << yellow_points->points.size()
         << "\n\tGreen points:\t" << green_points->points.size()
         << "\n\tCyan points:\t" << cyan_points->points.size()
         << "\n\tBlue points:\t" << blue_points->points.size() << std::endl;
    summary_file << "\nPoints assortment:"
                    "\n\tRed points:\t" << red_points->points.size()
                 << "\n\tOrange points:\t" << orange_points->points.size()
                 << "\n\tYellow points:\t" << yellow_points->points.size()
                 << "\n\tGreen points:\t" << green_points->points.size()
                 << "\n\tCyan points:\t" << cyan_points->points.size()
                 << "\n\tBlue points:\t" << blue_points->points.size() << std::endl;
    if ((red_max != 0) && (red_points->points.empty()))
        std::cout << "\nABNORMAL HEAT-MAP BEHAVIOUR!" << std::endl;
    // Exporting the points with a heat value
    overlapping_areas->width = overlapping_areas->points.size();
    overlapping_areas->height = 1;
    overlapping_areas->is_dense = false;
    pcl::io::savePCDFileASCII(file_base_name.str() + "_OV.pcd", *overlapping_areas);
    // Visualize the heat std::map point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("heat_map"));
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    vs->addPointCloud(blue_points, "blue");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "blue");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "blue");
    vs->addPointCloud(cyan_points, "cyan");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "cyan");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "cyan");
    vs->addPointCloud(green_points, "green");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "green");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "green");
    vs->addPointCloud(yellow_points, "yellow");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "yellow");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "yellow");
    vs->addPointCloud(orange_points, "orange");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 140.0/255.0, 0.0, "orange");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "orange");
    vs->addPointCloud(red_points, "red");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "red");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.5, "red");
    vs->resetCamera();
    vs->registerKeyboardCallback(&KeyboardEventOccurred, (void*) nullptr);
    while (!vs->wasStopped()) {
        vs->spinOnce(100);
        if (close_visualizer) {
            close_visualizer = false;
            vs->close();
        }
    }
}


void HeatMapVisualizerPOV(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<std::vector<int>> &subsets_points_idx,
                      std::stringstream &file_base_name, std::ofstream &summary_file) {
    std::map<int, int> point_heat_map;
    // Set each point heat value as 0
    for (unsigned int p_id = 0; p_id < cloud->points.size(); ++p_id)
        point_heat_map[p_id] = 0;
    // Set heat value
    for (auto & ss_i : subsets_points_idx) {
        for (int p_i : ss_i) {
            auto it = point_heat_map.find(p_i);
            if (it != point_heat_map.end())
                point_heat_map[p_i] ++;
        }
    }
    // Get the max heat and print points data
    std::stringstream point_data_file;
    point_data_file << file_base_name.str() << "_POINTS_HEAT.csv";
    std::ofstream point_data_file_header(point_data_file.str());
    point_data_file_header << "point, heat" << std::endl;
    std::ofstream output_f(point_data_file.str(), std::ios_base::app | std::ios_base::out);
    int max_votes = -std::numeric_limits<int>::max();
    for (auto & p_id : point_heat_map) {
        if (p_id.second > max_votes)
            max_votes = p_id.second;
        output_f << p_id.first << "," << p_id.second << std::endl;
    }
    // Set the heat ranges
    float heat_step = (float)max_votes / 6;
    float blue_max = 0 + heat_step;
    float cyan_max = blue_max + heat_step;
    float green_max = cyan_max + heat_step;
    float yellow_max = green_max + heat_step;
    float orange_max = yellow_max + heat_step;
    float red_max = orange_max + heat_step;
    std::cout << "\nHeat scale:"
            "\n\tRed\t\t...\t(" << orange_max << ", " << red_max << "]"
            "\n\tOrange\t...\t(" << yellow_max << ", " << orange_max << "]"
            "\n\tYellow\t...\t(" << green_max << ", " << yellow_max << "]"
            "\n\tGreen\t...\t(" << cyan_max << ", " << green_max << "]"
            "\n\tCyan\t...\t(" << blue_max << ", " << cyan_max << "]"
            "\n\tBlue\t...\t[" << 0 << ", " << blue_max << "]" << std::endl;
    summary_file << "\nHeat scale:"
                    "\n\tRed\t...\t(" << orange_max << ", " << red_max << "]"
                    "\n\tOrange\t...\t(" << yellow_max << ", " << orange_max << "]"
                    "\n\tYellow\t...\t(" << green_max << ", " << yellow_max << "]"
                    "\n\tGreen\t...\t(" << cyan_max << ", " << green_max << "]"
                    "\n\tCyan\t...\t(" << blue_max << ", " << cyan_max << "]"
                    "\n\tBlue\t...\t[" << 0 << ", " << blue_max << "]" << std::endl;
    // Build the point clouds according to their heat values
    pcl::PointCloud<pcl::PointXYZ>::Ptr blue_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cyan_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr green_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr yellow_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr orange_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr red_points (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr overlapping_areas (new pcl::PointCloud<pcl::PointXYZ> ());
    for (auto & p_id : point_heat_map) {
        if ((float)p_id.second <= blue_max)
            blue_points->points.push_back(cloud->points[p_id.first]);
        else if (((float)p_id.second > blue_max) && ((float)p_id.second <= cyan_max)) {
            cyan_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > cyan_max) && ((float)p_id.second <= green_max)) {
            green_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > green_max) && ((float)p_id.second <= yellow_max)) {
            yellow_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > yellow_max) && ((float)p_id.second <= orange_max)) {
            orange_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        } else if (((float)p_id.second > orange_max) && ((float)p_id.second <= red_max)) {
            red_points->points.push_back(cloud->points[p_id.first]);
            overlapping_areas->points.push_back(cloud->points[p_id.first]);
        }
    }
    std::cout << "\nPoints assortment:"
            "\n\tRed points:\t" << red_points->points.size()
         << "\n\tOrange points:\t" << orange_points->points.size()
         << "\n\tYellow points:\t" << yellow_points->points.size()
         << "\n\tGreen points:\t" << green_points->points.size()
         << "\n\tCyan points:\t" << cyan_points->points.size()
         << "\n\tBlue points:\t" << blue_points->points.size() << std::endl;
    summary_file << "\nPoints assortment:"
                    "\n\tRed points:\t" << red_points->points.size()
                 << "\n\tOrange points:\t" << orange_points->points.size()
                 << "\n\tYellow points:\t" << yellow_points->points.size()
                 << "\n\tGreen points:\t" << green_points->points.size()
                 << "\n\tCyan points:\t" << cyan_points->points.size()
                 << "\n\tBlue points:\t" << blue_points->points.size() << std::endl;
    if ((red_max != 0) && (red_points->points.empty()))
        std::cout << "\nABNORMAL HEAT-MAP BEHAVIOUR!" << std::endl;
    // Exporting the points with a heat value
    overlapping_areas->width = overlapping_areas->points.size();
    overlapping_areas->height = 1;
    overlapping_areas->is_dense = false;
    pcl::io::savePCDFileASCII(file_base_name.str() + "_OV.pcd", *overlapping_areas);
    // Visualize the heat std::map point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("heat_map"));
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    vs->addCoordinateSystem(0.01f, "reference");
    vs->initCameraParameters();
    vs->setCameraFieldOfView(0.20);
    vs->setCameraClipDistances(0.00, 50.0);
    vs->addPointCloud(blue_points, "blue");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "blue");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "blue");
    vs->addPointCloud(cyan_points, "cyan");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "cyan");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "cyan");
    vs->addPointCloud(green_points, "green");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "green");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "green");
    vs->addPointCloud(yellow_points, "yellow");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "yellow");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "yellow");
    vs->addPointCloud(orange_points, "orange");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 140.0/255.0, 0.0, "orange");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "orange");
    vs->addPointCloud(red_points, "red");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "red");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.5, "red");
    std::stringstream pos_name, scn_shot_name;
    std::vector<std::vector<float>> cam_positions;
    cam_positions.resize(3);
    cam_positions[0] = {0.0, 0.0, 0.0};
    cam_positions[1] = {180.0, 0.0, 0.0};
    cam_positions[2] = {0.0, 180.0, 0.0};
    for (unsigned int p_i = 0; p_i < cam_positions.size(); ++p_i) {
        pos_name.str("");
        scn_shot_name.str("");
        switch (p_i) {
            case 0 :
                pos_name << "FRONT_VIEW";
                break;
            case 1 :
                pos_name << "SIDE_VIEW";
                break;
            case 2 :
                pos_name << "TOP_VIEW";
                break;
            default:
                assert(false);
        }
        scn_shot_name << file_base_name.str() << "_HEATMAP_" << pos_name.str() << ".png";
        vs->setCameraPosition(0.0, 0.0, 30.0,
                              cam_positions[p_i][0] + 1, cam_positions[p_i][1] + 1, cam_positions[p_i][2],
                              0.0, 1.0, 0.0);
        vs->resetCamera();
        vs->spinOnce(100);
        vs->saveScreenshot(scn_shot_name.str());
    }
    vs->close();
}


void ComputeDensity(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::map<int, double> &density_values, double &max_density,
                    double &min_density, float radius) {
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
#pragma omp parallel for firstprivate(cloud)
    for (unsigned int p_i = 0; p_i < cloud->points.size(); ++p_i) {
        std::vector<int> nearest_ns;
        std::vector<float> nearest_ns_sqr_distances;
        tree.radiusSearch(cloud->points[p_i], radius, nearest_ns, nearest_ns_sqr_distances);
#pragma omp critical
        {
            density_values[p_i] = !nearest_ns.empty() ? (double)nearest_ns.size() : 0.0;
            std::cout << "\t[" << p_i << "] = " << density_values[p_i] << std::endl;
        }
    }
    max_density = -std::numeric_limits<double>::max();
    min_density = std::numeric_limits<double>::max();
    for (auto & p_id : density_values) {
        if (p_id.second > max_density)
            max_density = p_id.second;
        else if (p_id.second < min_density)
            min_density = p_id.second;
    }
}


double ComputeArea(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float radius) {
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    double total_area = 0;
    for (auto point : cloud->points) {
        std::vector<int> nearest_ns;
        std::vector<float> nearest_ns_sqr_distances;
        tree.radiusSearch(point, radius, nearest_ns, nearest_ns_sqr_distances);
        if (!nearest_ns.empty())
            total_area += (M_PI * pow(radius, 2)) / (double)nearest_ns.size();
    }
    return total_area;
}