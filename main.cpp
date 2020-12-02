#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <gflags/gflags.h>
#include <Python.h>

#include"cnpy.h"

#include "SemanticVoxelCloud.h"

#include <chrono>

#define CMAP_ROW 256
#define CMAP_COL 3

DEFINE_string(input_path, "/home/lincoln/Documents/dataset/out-recording49.sens/",
              "The path of input images");

DEFINE_string(sem_input_path, "/home/lincoln/Documents/dataset/semantic/",
              "The path of input images");

DEFINE_string(output_path, "/home/lincoln/Documents/dataset/output/",
              "The path of output point clouds");

DEFINE_string(net_name, "rf_lw101_nyu",
              "The Net for semantic segmentation");

DEFINE_string(frame_name, "frame-001245",
              "Prefix of the frame");

DEFINE_int32(frame_num,
             6780,
             "The total number of processing frames.");

DEFINE_int32(start_frame,
             0,
             "The starting frame index.");

DEFINE_int32(steps,
             50,
             "The steps for frames.");

DEFINE_double(lx,
             0.1,
             "Leaf size x");

DEFINE_double(ly,
              0.1,
              "Leaf size y");

DEFINE_double(lz,
              0.1,
              "Leaf size z");

DEFINE_bool(enable_rgb,
            false,
            "Enable RGB Point Cloud");

DEFINE_bool(enable_sem,
            true,
            "Enable Semantic Point Cloud");

DEFINE_bool(enable_sem_filter,
            true,
            "Enable Semantic Filtered Point Cloud");

using namespace std;

typedef Eigen::Matrix<float, 40, 1> Vector40f;

int argmax(Vector40f vector);

int main( int argc, char** argv ) {
    // gflags initialization
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // RGBD Image
    cv::Mat colorImg, depthImg;
    // Semantic Image
    cnpy::NpyArray semanImg;
    // Camera poses
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    cout << "converting images into pointclouds" << endl;

    // Camera Calibration
    double fx = 366.41;
    double fy = 366.41;
    double cx = 251.454;
    double cy = 205.593;
    double depthScale = 1000.0;

    // defines pcl type
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // init new pointcloud
    PointCloud::Ptr RGBPointCloud(new PointCloud);
    PointCloud::Ptr RGBPointCloudFiltered(new PointCloud);
    PointCloud::Ptr SemPointCloud(new PointCloud);
    PointCloud::Ptr SemPointCloudFiltered(new PointCloud);

    std::chrono::steady_clock::time_point t1, t2;

    // Construct RGB Point Cloud and Filtered Point Cloud
    if (FLAGS_enable_rgb)
    {
        t1 = std::chrono::steady_clock::now();
        for (int frame = FLAGS_start_frame; frame < FLAGS_frame_num; frame += FLAGS_steps) {
            if (frame % 100 == 0)
                cout << "RGB: " << frame << endl;
            // From frame index to file name prefix
            char frame_string[12];
            sprintf(frame_string, "frame-%06d", frame);
            // Camera Pose Loading
            ifstream fin(FLAGS_input_path + frame_string + ".pose.txt");
            if (!fin) {
                cerr << "Pose txt excluded!" << endl;
                return 1;
            }
            // Transfer pose from array to Transformation Matrix
            double data[13] = {0};
            for (auto &d:data) {
                fin >> d;
            }
            // Error Pose
            bool is_false_pose = false;
            if(data[0] == -1)
            {
                is_false_pose = true;
                for(int i = 1; i < 13; i++)
                {
                    if(data[i] != 0)
                    {
                        is_false_pose = false;
                    }
                }
            }
            if(is_false_pose)
            {
                continue;
            }
            Eigen::Matrix3d rotMatrix;
            rotMatrix << data[0], data[1], data[2],
                    data[4], data[5], data[6],
                    data[8], data[9], data[10];
            Eigen::Isometry3d T(rotMatrix);
            T.pretranslate(Eigen::Vector3d(data[3], data[7], data[11]));
            // Individual RGBD Image Frame Loading
            colorImg = cv::imread(FLAGS_input_path + frame_string + ".color.jpg");
            depthImg = cv::imread(FLAGS_input_path + frame_string + ".depth.pgm", -1);
            // Point Loading
            for (int v = 0; v < depthImg.rows; v++)
                for (int u = 0; u < depthImg.cols; u++) {
                    ushort d = depthImg.ptr<ushort>(v)[u];
                    if (d == 0)
                        continue;
                    Eigen::Vector3d point;
                    point[2] = double(d) / depthScale; // depth
                    point[0] = (u - cx) * point[2] / fx;
                    point[1] = (v - cy) * point[2] / fy;
                    Eigen::Vector3d pointWorld = T * point;

                    PointT p_rgb;
                    p_rgb.x = pointWorld[0];
                    p_rgb.y = pointWorld[1];
                    p_rgb.z = pointWorld[2];
                    p_rgb.y = -p_rgb.y;
                    p_rgb.z = -p_rgb.z;
                    p_rgb.b = colorImg.data[v * colorImg.step + u * colorImg.channels()];
                    p_rgb.g = colorImg.data[v * colorImg.step + u * colorImg.channels() + 1];
                    p_rgb.r = colorImg.data[v * colorImg.step + u * colorImg.channels() + 2];
                    RGBPointCloud->points.push_back(p_rgb);

                }
        }
        t2 = std::chrono::steady_clock::now();
        cout << "RGB Point Cloud Construction Finished: "
             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
             << "us" << endl;

        // Output Point Cloud
        RGBPointCloud->is_dense = true;
        cout << "Total points in RGB pointcloud: " << RGBPointCloud->size() << endl;
        pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".rgb.pcd", *RGBPointCloud);

        // Filter
        t1 = std::chrono::steady_clock::now();
        pcl::VoxelGrid<pcl::PointXYZRGB> vg_rgb;
        vg_rgb.setInputCloud(RGBPointCloud);
        vg_rgb.setLeafSize(FLAGS_lx, FLAGS_ly, FLAGS_lz);//change leaf size into 0.5cm
        vg_rgb.setSaveLeafLayout(true);
        vg_rgb.filter(*RGBPointCloudFiltered);
        t2 = std::chrono::steady_clock::now();
        cout << "RGB Filter Finished: "
             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
             << "us" << endl;

        // Output Point Cloud
        cout << "RGBPointCloud after filtering has: " << RGBPointCloudFiltered->points.size()
                  << " data points." << std::endl;
        pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".rgb.filtered.pcd",
                                   *RGBPointCloudFiltered);
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////
    // Construct Semantic Point Cloud and Filtered Point cloud
    // Vector for transformation between semantic class to corresponding BGR color
    cnpy::NpyArray cmap = cnpy::npy_load("../cmap.npy");
    uint8_t *cmap_data = cmap.data<uint8_t>();

    if (FLAGS_enable_sem)
    {
        t1 = std::chrono::steady_clock::now();
        for (int frame = FLAGS_start_frame; frame < FLAGS_frame_num; frame += FLAGS_steps) {
            if (frame % 100 == 0)
                cout << "Semantic: " << frame << endl;
            char frame_string[13];
            sprintf(frame_string, "frame-%06d", frame);

            ifstream fin(FLAGS_input_path + frame_string + ".pose.txt");
            if (!fin) {
                cerr << "Pose txt excluded!" << endl;
                return 1;
            }

            double data[12] = {0};
            for (auto &d:data) {
                fin >> d;
            }
            // Error Pose
            bool is_false_pose = false;
            if(data[0] == -1)
            {
                is_false_pose = true;
                for(int i = 1; i < 13; i++)
                {
                    if(data[i] != 0)
                    {
                        is_false_pose = false;
                    }
                }
            }
            if(is_false_pose)
            {
                continue;
            }
            Eigen::Matrix3d rotMatrix;
            rotMatrix << data[0], data[1], data[2],
                    data[4], data[5], data[6],
                    data[8], data[9], data[10];
            Eigen::Isometry3d T(rotMatrix);
            T.pretranslate(Eigen::Vector3d(data[3], data[7], data[11]));

            depthImg = cv::imread(FLAGS_input_path + frame_string + ".depth.pgm", -1);
            // Semantic Vector Loading
            semanImg = cnpy::npy_load(FLAGS_sem_input_path + frame_string + ".npy");

            for (int v = 0; v < depthImg.rows; v++)
                for (int u = 0; u < depthImg.cols; u++) {
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    ushort d = depthImg.ptr<ushort>(v)[u];
                    if (d == 0)
                        continue;
                    Eigen::Vector3d point;
                    point[2] = double(d) / depthScale; // depth
                    point[0] = (u - cx) * point[2] / fx;
                    point[1] = (v - cy) * point[2] / fy;
                    Eigen::Vector3d pointWorld = T * point;

                    PointT p_sem;

                    Vector40f semanticVector;
                    for (int idx = 0; idx < NUM_OF_CLASSES; idx++) {
                        semanticVector(idx) = *(semanImg.data<float>() + (depthImg.cols * v + u) * NUM_OF_CLASSES +
                                                idx);
                    }

                    p_sem.x = pointWorld[0];
                    p_sem.y = pointWorld[1];
                    p_sem.z = pointWorld[2];
                    p_sem.y = -p_sem.y;
                    p_sem.z = -p_sem.z;

                    int class_index = argmax(semanticVector);
                    p_sem.b = cmap_data[class_index + 2 * CMAP_ROW];
                    p_sem.g = cmap_data[class_index + CMAP_ROW];
                    p_sem.r = cmap_data[class_index];
                    SemPointCloud->points.push_back(p_sem);
                }
        }
        t2 = std::chrono::steady_clock::now();
        cout << "Semantic Point Cloud Construction Finished: "
             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
             << "us" << endl;

        // Output Point Cloud
        SemPointCloud->is_dense = true;
        cout<<"Total points in Semantic pointcloud: "<<SemPointCloud->points.size()<<endl;
        pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".semantic.pcd", *SemPointCloud );
    }

    // Semantic Filter
    if(FLAGS_enable_sem_filter)
    {
        t1 = std::chrono::steady_clock::now();
        SemanticVoxelCloud SemVoxelCloud(SemPointCloud, SemPointCloudFiltered);
        SemVoxelCloud.setInputPath(FLAGS_input_path);
        SemVoxelCloud.setSemInputPath(FLAGS_sem_input_path);
        SemVoxelCloud.setStartFrame(FLAGS_start_frame);
        SemVoxelCloud.setFrameNum(FLAGS_frame_num);
        SemVoxelCloud.setSteps(FLAGS_steps);
        SemVoxelCloud.filter(FLAGS_lx,FLAGS_ly,FLAGS_lz,true);
        t2 = std::chrono::steady_clock::now();
        cout << "Semantic Filter Finished: "
             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
             << "us" << endl;

        // Output Point Cloud
        cout << "SemPointCloud after filtering has: " << SemPointCloudFiltered->points.size()
                  << " data points." << std::endl;
        pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".semantic.filtered.pcd",
                                   *SemPointCloudFiltered);
    }

    return 0;
}

/*
 * argmax: Return the index of the element which has the maximum probability in the semantic vector
 */
int argmax(Vector40f vector)
{
    float max = vector(0);
    int argmax = 0;
    for(int i = 1; i < NUM_OF_CLASSES; i++)
    {
        if(vector(i) > max)
        {
            max = vector(i);
            argmax = i;
        }
    }
    return argmax;
}