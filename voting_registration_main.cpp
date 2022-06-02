//
// Created by Luis A. Peralta M. on 20/09/09.
//
#include "tools.hpp"

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <chrono>
#include <thread>
#include <cmath>

// Supervoxel segmentation parameters
#define VOXEL_RESOLUTION 2.5f
#define COLOR_WEIGHT 0.0f
#define SPATIAL_WEIGHT 2.5f
#define NORMALS_WEIGHT 0.5f

// Other parameters
#define NORMALS_RADIUS 10.0f
#define LAMBDA 3.0f
#define PSI_THR 10E-2f


using std::cout;
using std::endl;
using std::cerr;


int main(int argc, char **argv) {
    // Loading the point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    std::vector<int> dummy;
    std::vector<int> clouds = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
    if (clouds.size() != 2) {
        cerr << "ERROR: Need 2 '.pcd' files!" << endl;
        return -1;
    }
    // Source cloud
    if (pcl::io::loadPCDFile(argv[clouds[0]], *source_cloud) == -1) {
        cerr << "ERROR: Could not load " <<  argv[1] << " as 'SOURCE' point cloud" << endl;
        return -1;
    }
    removeNaNFromPointCloud(*source_cloud, *source_cloud, dummy);
    cout << "\nFile " << argv[1] << " successfully loaded: " << source_cloud->size() << " points." << endl;
    // Target cloud
    if (pcl::io::loadPCDFile(argv[clouds[1]], *target_cloud) == -1) {
        
        cerr << "ERROR: Could not load " << argv[2] << " as 'TARGET' point cloud" << endl;
        return -1;
    }
    removeNaNFromPointCloud(*target_cloud, *target_cloud, dummy);
    cout << "File " << argv[2] << " successfully loaded: " << target_cloud->size() << " points." << endl;


    // Reading input parameters
    // Registration method
    //      1 ... ICP           2 ... N-ICP
    //      3 ... TrICP         4 ... LM-ICP
    int registration_option;
    pcl::console::parse_argument(argc, argv, "--reg_option", registration_option);

    // Seed radius option to define a running mode
    //      1 ... Defined by the user       2 ...From R_max to R_min
    int seed_radius_option;
    pcl::console::parse_argument(argc, argv, "--seed_option", seed_radius_option);

    // Number of bins in the accumulator
    int n_bins;
    std::string name_bins;
    pcl::console::parse_argument(argc, argv, "--n_bins", n_bins);
    name_bins = "Nb:" + std::to_string(n_bins);

    // Base axis of rotation for the subsets
    //      1 ... X       2 ... Y       3 ... Z
    int r_axis;
    pcl::console::parse_argument(argc, argv, "--base_axis", r_axis);

    // Number of rotations around the base axis
    int n_rotations;
    std::string name_rotations;
    pcl::console::parse_argument(argc, argv, "--rotations", n_rotations);
    name_rotations = "Nr:" + std::to_string(n_rotations);

    // A pause time for the visualizer in milliseconds (only used when debugging)
    int p_time;
    pcl::console::parse_argument(argc, argv, "--p_time", p_time);


    // Compute the cloud resolutions
    cout << "\nComputing the resolutions..." << endl;
    float source_resolution = ComputeCloudResolution(source_cloud);
    float target_resolution = ComputeCloudResolution(target_cloud);
    cout << "\tSource resolution = " << source_resolution
         << "\n\tTarget resolution = " << target_resolution << endl;


    // Start measuring the processing time
    auto start_process = std::chrono::high_resolution_clock::now();


    // Estimate translation limits
    cout << "\nEstimating translation limits..." << endl;
    pcl::PointXYZ source_min_point, source_max_point, target_min_point, target_max_point;
    boundingBox(source_cloud, source_min_point, source_max_point);
    boundingBox(target_cloud, target_min_point, target_max_point);
    float x_min_limit, x_max_limit, y_min_limit, y_max_limit, z_min_limit, z_max_limit;
    // MIN and MAX translation limits in X
    if (source_min_point.x < target_min_point.x)
        x_min_limit = source_min_point.x;
    else
        x_min_limit = target_min_point.x;
    if (source_max_point.x > target_max_point.x)
        x_max_limit = source_max_point.x;
    else
        x_max_limit = target_max_point.x;
    // MIN and MAX translation limits in Y
    if (source_min_point.y < target_min_point.y)
        y_min_limit = source_min_point.y;
    else
        y_min_limit = target_min_point.y;
    if (source_max_point.y > target_max_point.y)
        y_max_limit = source_max_point.y;
    else
        y_max_limit = target_max_point.y;
    // MIN and MAX translation limits in Z
    if (source_min_point.z < target_min_point.z)
        z_min_limit = source_min_point.z;
    else
        z_min_limit = target_min_point.z;
    if (source_max_point.z > target_max_point.z)
        z_max_limit = source_max_point.z;
    else
        z_max_limit = target_max_point.z;
    // Definitive limits
    double x_limit = x_max_limit - x_min_limit + 0.5 * (x_max_limit - x_min_limit);
    double y_limit = y_max_limit - y_min_limit + 0.5 * (y_max_limit - y_min_limit);
    double z_limit = z_max_limit - z_min_limit + 0.5 * (z_max_limit - z_min_limit);


    // Histogram bins
    double q_bin_range = (1.0 - -1.0) / n_bins;
    double t_X_bin_range = (x_limit - -x_limit) / n_bins;
    double t_Y_bin_range = (y_limit - -y_limit) / n_bins;
    double t_Z_bin_range = (z_limit - -z_limit) / n_bins;


    // Scaling parameters
    float s_voxel_resolution = VOXEL_RESOLUTION * source_resolution;
    float s_color_weight = COLOR_WEIGHT * source_resolution;
    float s_spatial_weight = SPATIAL_WEIGHT * source_resolution;
    float s_normals_weight = NORMALS_WEIGHT * source_resolution;
    float s_normals_radius = NORMALS_RADIUS * source_resolution;
    float t_normals_radius = NORMALS_RADIUS * target_resolution;
    float inlier_threshold = 2 * source_resolution;


    // Normals estimation
    // Source
    pcl::PointCloud<pcl::PointNormal>::Ptr source_normals (new pcl::PointCloud<pcl::PointNormal> ());
    ComputePointNormals(source_cloud, s_normals_radius, source_normals);
    // Target
    pcl::PointCloud<pcl::PointNormal>::Ptr target_normals (new pcl::PointCloud<pcl::PointNormal> ());
    ComputePointNormals(target_cloud, t_normals_radius, target_normals);


    // Setting running option
    int seed_radius, max_seed_radius, min_seed_radius;
    std::string seed_name;
    switch (seed_radius_option) {
        case 1 : {
            // R_seed defined by the user
            pcl::console::parse_argument(argc, argv, "--seed_radius", seed_radius);
            min_seed_radius = seed_radius - 1;
            seed_name = std::to_string(seed_radius) + "CR";
        }
            break;
        case 2 : {
            // Defining MAX & MIN SEED RADIUS
            cout << "\nEstimating maximum and minimum seed radius..." << endl;
            // Source bounding box
            double source_bbox_size = std::sqrt(pow(source_max_point.x - source_min_point.x, 2) +
                                           pow(source_max_point.y - source_min_point.y, 2) +
                                           pow(source_max_point.z - source_min_point.z, 2));
            // Target bounding box
            double target_bbox_size = std::sqrt(pow(target_max_point.x - target_min_point.x, 2) +
                                           pow(target_max_point.y - target_min_point.y, 2) +
                                           pow(target_max_point.z - target_min_point.z, 2));
            // Bounding boxes sizes in CR
            float source_bbox_size_cr = (float) source_bbox_size / source_resolution;
            float target_bbox_size_cr = (float) target_bbox_size / target_resolution;
            // max_seed_radius and min_seed_radius
            float mean_box_size = (source_bbox_size_cr + target_bbox_size_cr) / 2;
            max_seed_radius = (int)round(mean_box_size / 2);
            if (max_seed_radius > 1000) {
                max_seed_radius = 1000;
                min_seed_radius = (int)round(100 * VOXEL_RESOLUTION);
            }
            else
                min_seed_radius = (int)round(10 * VOXEL_RESOLUTION);
            seed_radius = max_seed_radius;
            std::string max_name = std::to_string(max_seed_radius);
            std::string min_name = std::to_string(min_seed_radius);
            seed_name = max_name + "CR->" + min_name + "CR";
        }
            break;
        default :
            cerr << "\nERROR: No option available for 'seed radius option' was defined!"
                    "\nPlease define an available option:\n"
                    "\t1 ... Evaluate one single value of 'seed radius'\n"
                    "\t2 ... Evaluate several values of 'seed radius' from a MAX to a MIN" << endl;
            return -1;
    }


    // Setting visualizer
    pcl::PointCloud<pcl::PointXYZL>::Ptr source_cloud_sv (new pcl::PointCloud<pcl::PointXYZL> ());
    int v_1(0), v_2(0);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vs (new pcl::visualization::PCLVisualizer ("process_visualization"));
    vs->createViewPort(0.0, 0.0, 0.5, 1.0, v_1);
    vs->createViewPort(0.5, 0.0, 1.0, 1.0, v_2);
    vs->setBackgroundColor(1.0, 1.0, 1.0);
    // Text
    std::stringstream c_seed_r, c_subset;
    c_seed_r << "Seed radius: " << 0;
    c_subset << "Subset: " << 0;
    vs->addText(c_seed_r.str(), 10, 10, 15, 0.0, 0.0, 0.0, "seed_radius_id", v_1);
    vs->addText(c_subset.str(), 10, 10, 15, 0.0, 0.0, 0.0, "subset_id", v_2);
    // Segmented source in view port 1
    vs->addPointCloud(source_cloud_sv, "segmentation", v_1);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "segmentation");
    // Source subset in view port 1
    vs->addPointCloud(source_cloud, "subset_v_1", v_1);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "subset_v_1");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "subset_v_1");
    // Source in view port 2
    vs->addPointCloud(source_cloud,"source_v_2", v_2);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_v_2");
    // Target in view port 2
    vs->addPointCloud(target_cloud,"target_v_2", v_2);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "target_v_2");
    // Source subset in view port 2
    vs->addPointCloud(source_cloud, "subset_v_2", v_2);
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "subset_v_2");
    vs->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "subset_v_2");
    vs->resetCamera();
    vs->registerKeyboardCallback(&KeyboardEventOccurred, (void*) nullptr);



    // Defining the rotation angles
    float angle_step = 360.0f / (float)n_rotations;
    float c_angle = angle_step;
    std::vector<float> rotation_angles;
    for (int r_i = 0; r_i < n_rotations; ++r_i) {
        rotation_angles.push_back(c_angle);
        c_angle += angle_step;
    }

    // Add a translation to the datasets
    Eigen::Affine3f t_only_matrix = Eigen::Affine3f::Identity();
    t_only_matrix.translation() << 0 * source_resolution, 0 * source_resolution, 0 * source_resolution;
    pcl::transformPointCloud(*source_cloud, *source_cloud, t_only_matrix.inverse());

    // Defining Voting step
    int voting_step = n_bins / 10;


    // Main process
    accumulator_7D accumulator;
    w_vector_int aux_accumulator;
    int c_sv_number, p_sv_number = 0;
    float s_seed_radius;
    while(seed_radius > min_seed_radius) {
        // Current seed radius
        s_seed_radius = (float)seed_radius * source_resolution;
        c_seed_r.str("");
        c_seed_r << "Seed radius: " << seed_radius;

        // Segmentation of the source cloud
        std::vector<std::vector<int>> source_subsets_ids;
        SupervoxelSegmentation(s_voxel_resolution, s_seed_radius, s_color_weight, s_spatial_weight, s_normals_weight,
                               source_cloud, source_cloud_sv, source_subsets_ids);
        // Update visualizer
        vs->updateText(c_seed_r.str(), 10, 30, 15, 0.0, 0.0, 0.0, "seed_radius_id");
        vs->updatePointCloud(source_cloud_sv, "segmentation");
        // Build the subsets
        std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> source_normals_subsets;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> source_cloud_subsets;
        BuildSubsets(source_cloud, source_normals, source_subsets_ids, source_cloud_subsets, source_normals_subsets);

        // Check the number of subsets
        c_sv_number = (int)source_cloud_subsets.size();
        if (c_sv_number != p_sv_number) {
            // The number of supervoxels has changed. Thus, current R_seed is evaluated.
            p_sv_number = c_sv_number;

            // For all subsets of P
            for (unsigned int ss_i = 0; ss_i < source_cloud_subsets.size(); ++ss_i) {
                // Query source subset
                pcl::PointCloud<pcl::PointXYZ>::Ptr c_source_subset = source_cloud_subsets[ss_i];
                pcl::PointCloud<pcl::PointNormal>::Ptr c_source_n_subset = source_normals_subsets[ss_i];

                // Rotation and ICP
#pragma omp parallel for firstprivate(rotation_angles, c_source_subset, c_source_n_subset, source_cloud, target_cloud, target_normals, r_axis)
                for (unsigned int a_i = 0; a_i < rotation_angles.size(); ++a_i) {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr c_source_subset_tr (new pcl::PointCloud<pcl::PointXYZ>());
                    pcl::PointCloud<pcl::PointNormal>::Ptr c_source_n_subset_tr (new pcl::PointCloud<pcl::PointNormal>());
                    // Query rotation matrix
                    Eigen::Matrix4d c_rotation_only_matrix;
                    BuildRotationMatrix(rotation_angles[a_i], c_rotation_only_matrix, r_axis);
                    // Apply rotation
                    pcl::transformPointCloud(*c_source_subset, *c_source_subset_tr, c_rotation_only_matrix);
                    pcl::transformPointCloudWithNormals(*c_source_n_subset, *c_source_n_subset_tr, c_rotation_only_matrix);

                    // Core registration
                    Eigen::Matrix4d c_icp_matrix;
                    double c_fitness_score;
                    switch (registration_option) {
                        case 1 : // ICP
                            RunICP(c_source_subset_tr, target_cloud, c_icp_matrix, c_fitness_score);
                            break;
                        case 2 : // N-ICP
                            RunNICP(c_source_n_subset_tr, target_normals, c_icp_matrix, c_fitness_score);
                            break;
                        case 3 : // TrICP
                            RunTrICP(c_source_subset_tr, target_cloud, c_icp_matrix, c_fitness_score);
                            break;
                        case 4 : // LM-ICP
                            RunLMICP(c_source_subset_tr, target_cloud, c_icp_matrix, c_fitness_score);
                            break;
                        default : // ICP
                            RunICP(c_source_subset_tr, target_cloud, c_icp_matrix, c_fitness_score);
                            break;
                    }

                    // Proceeding with no faulty registrations
                    if (c_fitness_score != 100) {
                        Eigen::Matrix4d c_transformation_matrix = c_icp_matrix * c_rotation_only_matrix;
                        double c_psi, c_msep;
                        ComputeMetrics(source_cloud, target_cloud, c_transformation_matrix, inlier_threshold, LAMBDA,
                                       c_msep, c_psi);
                        if (isfinite(c_psi) && (c_psi < PSI_THR)) {
                            // Fitting the transformation into the accumulator
                            std::vector<double> c_transformation_vector;
                            MatrixToVector(c_transformation_matrix, c_transformation_vector);
                            int bin_q_W = (int)round((c_transformation_vector[0] - -1.0) / q_bin_range);
                            int bin_q_X = (int)round((c_transformation_vector[1] - -1.0) / q_bin_range);
                            int bin_q_Y = (int)round((c_transformation_vector[2] - -1.0) / q_bin_range);
                            int bin_q_Z = (int)round((c_transformation_vector[3] - -1.0) / q_bin_range);
                            int bin_t_X = (int)round((c_transformation_vector[4] - -x_limit) / t_X_bin_range);
                            int bin_t_Y = (int)round((c_transformation_vector[5] - -y_limit) / t_Y_bin_range);
                            int bin_t_Z = (int)round((c_transformation_vector[6] - -z_limit) / t_Z_bin_range);
                            // For visualization
                            pcl::PointCloud<pcl::PointXYZ>::Ptr vs_source_tr (new pcl::PointCloud<pcl::PointXYZ>());
                            pcl::PointCloud<pcl::PointXYZ>::Ptr vs_c_source_subset_tr (new pcl::PointCloud<pcl::PointXYZ>());
                            pcl::transformPointCloud(*source_cloud, *vs_source_tr, c_transformation_matrix);
                            pcl::transformPointCloud(*c_source_subset, *vs_c_source_subset_tr, c_transformation_matrix);
#pragma omp critical
                            {
                                // Binning/Voting
                                std::vector<int> subset_id {(int)ss_i, (int)ss_i};
                                Voting(subset_id, seed_radius, c_psi, rotation_angles[a_i], c_transformation_vector,
                                       bin_q_W, bin_q_X, bin_q_Y, bin_q_Z, bin_t_X, bin_t_Y, bin_t_Z, voting_step,
                                       accumulator, aux_accumulator);
                                // Verbosity
                                cout << "Seed radius: " << seed_radius << " CR\t|\tSubset: " << ss_i
                                     << "\t|\tAngle: " << rotation_angles[a_i] << "\t|\tPSI: " << c_psi << endl;
                                // Visualizer verbosity
                                c_subset.str("");
                                c_subset << "Subset: " << ss_i << " | Angle: " << rotation_angles[a_i];
                                vs->updateText(c_subset.str(), 10, 30, 15, 0.0, 0.0, 0.0, "subset_id");
                                vs->updatePointCloud(c_source_subset, "subset_v_1");
                                vs->updatePointCloud(vs_c_source_subset_tr, "subset_v_2");
                                vs->updatePointCloud(vs_source_tr, "source_v_2");
                                vs->spinOnce(p_time);
                            }
                        }
                        else
                            cout << "Seed radius: " << seed_radius << " CR\t|\tSubset: " << ss_i
                                 << "\t|\tAngle: " << rotation_angles[a_i] << "\t|\tPSI ERROR!" << endl;
                    }
                    else
                        cout << "Seed radius: " << seed_radius << " CR\t|\tSubset: " << ss_i
                             << "\t|\tAngle: " << rotation_angles[a_i] << "\t|\tFITNESS SCORE ERROR!" << endl;
                }
            }
        }
        else
            // The number of supervoxels hasn't changed. Thus, current R_seed is skipped.
            p_sv_number = c_sv_number;
        // Decreasing R_seed
        seed_radius--;
    }
    // Closing process visualizer.
    vs->close();
    if (accumulator.empty()) {
        cerr << "\nERROR: Accumulator empty!" << endl;
        return -1;
    }


    // Start measuring search time
    auto start_search = std::chrono::high_resolution_clock::now();


    // Votes counting and obtaining PSI and angles
    std::vector<int> votes_v;
    std::vector<double> psi_v;
    std::vector<float> r_angles_v;
    VotesCounting(accumulator, aux_accumulator, votes_v, psi_v, r_angles_v);


    // Reading the data set files names
    // Source
    std::string p_file = argv[1];
    size_t p_last_diagonal = p_file.find_last_of('/');
    size_t p_last_dot = p_file.find_last_of('.');
    std::string p_ds_name = p_file.substr(p_last_diagonal + 1, p_last_dot - p_last_diagonal - 1);
    // Target
    std::string q_file = argv[2];
    size_t q_last_diagonal = q_file.find_last_of('/');
    size_t q_last_dot = q_file.find_last_of('.');
    std::string q_ds_name = q_file.substr(q_last_diagonal + 1, q_last_dot - q_last_diagonal - 1);


    // Defining the registration name for data file
    std::string reg_name;
    switch (registration_option) {
        case 1:
            reg_name = "ICP";
            break;
        case 2:
            reg_name = "N-ICP";
            break;
        case 3:
            reg_name = "TrICP";
            break;
        case 4:
            reg_name = "LM-ICP";
            break;
        default:
            reg_name = "ICP";
            break;
    }


    // Defining the output base name
    std::stringstream output_base_name;
    output_base_name << "ALL_SS_" << p_ds_name << "->" << q_ds_name << "_" << reg_name << "_" << seed_name
                     << "_" << name_bins << "_" << name_rotations;


    // Summary file
    std::stringstream summary_file_name;
    summary_file_name << output_base_name.str() << "_SUMMARY.txt";
    ofstream summary_file;
    summary_file.open(summary_file_name.str());


    // Looking for most voted bin
    std::vector<int> best_bin_by_votes;
    std::vector<std::vector<double>> best_transformations_v_by_votes;
    std::vector<std::vector<int>> best_corresponding_subsets_by_votes;
    std::map<int, std::vector<int>> best_seeds_and_subsets_by_votes;
    FindBestBinByVotes(aux_accumulator, votes_v, psi_v, best_bin_by_votes, summary_file);
    GetBestBinTransformation(accumulator, best_bin_by_votes, best_transformations_v_by_votes,
                     best_corresponding_subsets_by_votes, best_seeds_and_subsets_by_votes);


    // Computing the average transformation from the most voted bin
    std::vector<double> avg_best_tr_v_by_votes;
    Eigen::Matrix4d avg_best_tr_m_by_votes;
    ComputeAverageTransformation(best_transformations_v_by_votes, avg_best_tr_v_by_votes);
    VectorToMatrix(avg_best_tr_v_by_votes, avg_best_tr_m_by_votes);
    if (avg_best_tr_v_by_votes.empty()) {
        cerr << "ERROR: Average T does not exist!" << endl;
        return -1;
    }
    cout << "\tQw, Qx, Qy, Qz, Tx, Ty, Tz:\n\t";
    summary_file << "\tQw, Qx, Qy, Qz, Tx, Ty, Tz:\n\t";
    for (unsigned int i = 0; i < avg_best_tr_v_by_votes.size(); ++i) {
        if (i == (avg_best_tr_v_by_votes.size() - 1)) {
            cout << avg_best_tr_v_by_votes[i] << endl;
            summary_file << avg_best_tr_v_by_votes[i] << endl;
        }
        else {
            cout << avg_best_tr_v_by_votes[i] << ", ";
            summary_file << avg_best_tr_v_by_votes[i] << ", ";
        }
    }


    // Measuring the total processing time
    auto stop_search = std::chrono::high_resolution_clock::now();
    auto stop_process = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_search - start_search);
    auto process_duration_s = std::chrono::duration_cast<std::chrono::seconds>(stop_process - start_process);
    auto process_duration_m = std::chrono::duration_cast<std::chrono::minutes>(stop_process - start_process);
    auto process_duration_h = std::chrono::duration_cast<std::chrono::hours>(stop_process - start_process);


    // Print out the binning on terminal and data file
    std::stringstream transformations_data_file;
    transformations_data_file << output_base_name.str() << "_BINS_DATA.csv";
    ofstream transformations_data_file_header (transformations_data_file.str());
    transformations_data_file_header << "No, bQw, bQx, bQy, bQz, bTx, bTy, bTz, votes, angle, psi" << endl;
    PrintAccumulatorAndMetrics(aux_accumulator, votes_v, psi_v, r_angles_v, transformations_data_file);


    // Printing out the time in the console and summary file
    cout << "\nTransformation search finished in " << search_duration.count() << " microseconds." << endl;
    summary_file << "\nTransformation search finished in " << search_duration.count() << " microseconds." << endl;
    cout << "\nProcess finished in ";
    summary_file << "\nProcess finished in ";
    if (process_duration_s.count() <= 60) {
        cout << process_duration_s.count() << " seconds." << endl;
        summary_file << process_duration_s.count() << " seconds." << endl;
    }
    else if ((process_duration_s.count() > 60) && (process_duration_m.count() <= 60)) {
        cout << process_duration_m.count() << " minutes." << endl;
        summary_file << process_duration_m.count() << " minutes." << endl;
    }
    else if ((process_duration_m.count() > 60) && (process_duration_h.count() <= 24)) {
        cout << process_duration_h.count() << " hours." << endl;
        summary_file << process_duration_h.count() << " hours." << endl;
    }


    // Extracting the origin subsets as point idx
    cout << "\nExtracting information for visualization and heat-map..." << endl;
    std::vector<std::vector<int>> best_subsets_points_ids;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> best_corresponding_subsets_clouds;
    best_corresponding_subsets_clouds.resize(best_corresponding_subsets_by_votes.size());
    pcl::PointCloud<pcl::PointXYZL>::Ptr query_sv_cloud (new pcl::PointCloud<pcl::PointXYZL> ());
    for (auto & it : best_seeds_and_subsets_by_votes) {
        int query_seed = it.first;
        std::vector<int> query_subsets_ids = it.second;
        query_sv_cloud->clear();
        std::vector<std::vector<int>> query_sv_ids;
        SupervoxelSegmentation(s_voxel_resolution, (float)query_seed * source_resolution, s_color_weight,
                               s_spatial_weight, s_normals_weight, source_cloud, query_sv_cloud, query_sv_ids);
        for (auto & q_ss_i : query_subsets_ids)
            best_subsets_points_ids.push_back(query_sv_ids[q_ss_i]);
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> query_sv_clouds;
        std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> query_sv_normals;
        BuildSubsets(source_cloud, source_normals, query_sv_ids, query_sv_clouds, query_sv_normals);
        for (unsigned int i = 0; i < best_corresponding_subsets_by_votes.size(); ++i) {
            int query_c_seed = best_corresponding_subsets_by_votes[i][2];
            int query_c_subset = best_corresponding_subsets_by_votes[i][0];
            if (query_seed == query_c_seed)
                best_corresponding_subsets_clouds[i] = query_sv_clouds[query_c_subset];
        }
    }


    // Visualizing leading transformations
    // Visualizing the best transformations with overlap and source's heatmap
    double avg_psi, avg_msep;
    ComputeMetrics(source_cloud, target_cloud, avg_best_tr_m_by_votes, inlier_threshold, LAMBDA,
                   avg_msep, avg_psi);
    cout << "\nBest transformation by votes PSI = " << avg_psi << endl;
    cout << "\nBest transformation by votes MSEp = " << avg_msep << endl;
    summary_file << "\nBest transformation by votes PSI = " << avg_psi << endl;
    summary_file << "\nBest transformation by votes MSEp = " << avg_msep << endl;
    StaticVisualizerPOV(avg_best_tr_m_by_votes, source_cloud, target_cloud, inlier_threshold, output_base_name);
    HeatMapVisualizerPOV(source_cloud, best_subsets_points_ids, output_base_name, summary_file);
//    StaticVisualizer(avg_best_tr_m_by_votes, source_cloud, target_cloud, inlier_threshold);
//    HeatMapVisualizer(source_cloud, best_subsets_points_ids, output_base_name, summary_file);


    summary_file.close();
    return 0;
}