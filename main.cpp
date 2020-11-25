#include <iostream>
#include <cstdlib>
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
#include <cmath>
#include <cstring>
#include <gflags/gflags.h>
#include <Python.h>
//#include <numpy/arrayobject.h>

//#include "NumCpp.hpp"

#include"cnpy.h"

#include "SemanticVoxelCloud.h"
#include "ProcessVector.h"

#include <chrono>

#define NUM_OF_CLASS 40

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

DEFINE_bool(enable_rgb,
            false,
            "Enable RGB Point Cloud");

DEFINE_bool(enable_sem,
            true,
            "Enable Semantic Point Cloud");

using namespace std;

int argmax(float* vector);

int main( int argc, char** argv ) {
    // gflags initialization
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // RGBD Image
    cv::Mat colorImg, depthImg;
    cv::Mat semanImg;
    // Semantic Vectors
    cnpy::NpyArray semanticVectors;
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

    // Py_Initialize();

    // Construct RGB Point Cloud and Filtered Point Cloud
    for (int frame = FLAGS_start_frame; frame < FLAGS_frame_num; frame += FLAGS_steps) {
        if (frame % 100 == 0)
            cout << "RGB: " << frame << endl;
        // From frame index to file name prefix
        char frame_string[7];
        sprintf(frame_string, "frame-%06d", frame);
        // Individual RGBD Image Frame Loading
        colorImg = cv::imread(FLAGS_input_path + frame_string + ".color.jpg");
        depthImg = cv::imread(FLAGS_input_path + frame_string + ".depth.pgm", -1);
        // Camera Pose Loading
        ifstream fin(FLAGS_input_path + frame_string + ".pose.txt");
        if (!fin) {
            cerr << "Pose txt excluded!" << endl;
            return 1;
        }
        // Transfer pose from array to Transformation Matrix
        double data[12] = {0};
        for (auto &d:data) {
            fin >> d;
        }
        Eigen::Matrix3d rotMatrix;
        rotMatrix << data[0], data[1], data[2],
                data[4], data[5], data[6],
                data[8], data[9], data[10];
        Eigen::Isometry3d T(rotMatrix);
        T.pretranslate(Eigen::Vector3d(data[3], data[7], data[11]));
        // Point Loading
        for (int v = 0; v < colorImg.rows; v++)
            for (int u = 0; u < colorImg.cols; u++) {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

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
                if (p_rgb.b != 0 || p_rgb.g != 0 || p_rgb.r != 0)
                    RGBPointCloud->points.push_back(p_rgb);
            }

        // Filter
        pcl::VoxelGrid<pcl::PointXYZRGB> vg_rgb;
        vg_rgb.setInputCloud(RGBPointCloud);
        vg_rgb.setLeafSize(0.01f, 0.01f, 0.01f);//change leaf size into 0.5cm
        vg_rgb.filter(*RGBPointCloudFiltered);

        // Construct Semantic Point Cloud and Filtered Point cloud
        // Vector for transformation between semantic class to corresponding BGR color
        cnpy::NpyArray cmap = cnpy::npy_load(FLAGS_sem_input_path + frame_string + ".npy");
        uint8_t *cmap_data = cmap.data<uint8_t>();
        Eigen::Vector3f leaf_size(0.01f, 0.01f, 0.01f);
        SemanticVoxelCloud SemVoxelCloud(leaf_size, 0);
        size_t* voxel_index = (std::size_t*) calloc(vg_rgb.getLeafLayout().size(), sizeof(std::size_t));
        for (int frame = FLAGS_start_frame; frame < FLAGS_frame_num; frame += FLAGS_steps) {
            if (frame % 100 == 0)
                cout << "Semantic: " << frame << endl;
            char frame_string[7];
            sprintf(frame_string, "%06d", frame);
            depthImg = cv::imread(FLAGS_input_path + "frame-" + frame_string + ".depth.pgm", -1);
            semanticVectors = cnpy::npy_load(FLAGS_sem_input_path + "frame-" + frame_string + ".npy");

            ifstream fin(FLAGS_input_path + "frame-" + frame_string + ".pose.txt");
            if (!fin) {
                cerr << "Pose txt excluded!" << endl;
                return 1;
            }

            double data[12] = {0};
            for (auto &d:data) {
                fin >> d;
            }
            Eigen::Matrix3d rotMatrix;
            rotMatrix << data[0], data[1], data[2],
                    data[4], data[5], data[6],
                    data[8], data[9], data[10];
            Eigen::Isometry3d T(rotMatrix);
            T.pretranslate(Eigen::Vector3d(data[3], data[7], data[11]));

            for (int v = 0; v < colorImg.rows; v++)
                for (int u = 0; u < colorImg.cols; u++) {
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

                    if (FLAGS_enable_sem) {
                        float *semanticVector = (semanticVectors.data<float>() +
                                                 (colorImg.cols * v + u) * NUM_OF_CLASSES);

                        p_sem.x = pointWorld[0];
                        p_sem.y = pointWorld[1];
                        p_sem.z = pointWorld[2];
                        p_sem.y = -p_sem.y;
                        p_sem.z = -p_sem.z;

                        int class_index = argmax(semanticVector);
                        p_sem.b = cmap_data[class_index];
                        p_sem.g = cmap_data[class_index + 1];
                        p_sem.r = cmap_data[class_index + 2];
                        if(p_sem.b != 0 || p_sem.g != 0 || p_sem.r !=0)
                            SemPointCloud->points.push_back( p_sem );

//                        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
//                        SemVoxelCloud.addPoint(p_sem.x, p_sem.y, p_sem.z, semanticVector);
//                        std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
//                        cout << "addPoint: "
//                             << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count()
//                             << "us.\n";
//
//                        std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
//                        voxel_index[vg_rgb.getCentroidIndexAt(vg_rgb.getGridCoordinates(p_sem.x, p_sem.y, p_sem.z))] = SemVoxelCloud.getNumberOfGrids();
//                        std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();
//                        cout << "Voxel Index: "
//                             << std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count()
//                             << "us." << endl;
                    }

                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                    cout << frame << " " << v << " " << u << ": "
                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
                         << "us." << endl << endl;
                }
        }

        // Output Point Cloud
        if (FLAGS_enable_rgb) {
            RGBPointCloud->is_dense = true;
            cout << "Total points in RGB pointcloud: " << RGBPointCloud->size() << endl;
            pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".rgb.pcd", *RGBPointCloud);

            std::cout << "RGBPointCloud after filtering has: " << RGBPointCloudFiltered->points.size()
                      << " data points." << std::endl;
            pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".rgb.filtered.pcd",
                                       *RGBPointCloudFiltered);
        }
        if (FLAGS_enable_sem) {
            SemPointCloud->is_dense = true;
            cout<<"Total points in Semantic pointcloud: "<<SemPointCloud->size()<<endl;
            pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".semantic.pcd", *SemPointCloud );

            /*//pcl::MedianFilter<pcl::PointXYZRGB> vg_sem;
            pcl::VoxelGrid<pcl::PointXYZRGB> vg_sem;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr SemPointCloudFiltered (new pcl::PointCloud<pcl::PointXYZRGB>);
            vg_sem.setInputCloud (SemPointCloud);
            //vg_sem.setMaxAllowedMovement(10);
            vg_sem.setLeafSize (0.1f, 0.1f, 0.1f);//change leaf size into 0.5cm
            //vg_sem.setWindowSize(10);
            vg_sem.filter (*SemPointCloudFiltered);
            std::cout << "SemPointCloud after filtering has: " << SemPointCloudFiltered->points.size ()  << " data points." << std::endl;
            pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".semantic.filtered.pcd", *SemPointCloudFiltered );*/

//            std::chrono::steady_clock::time_point t9 = std::chrono::steady_clock::now();
//            SemPointCloud->is_dense = true;
//            for (int i = 0; i < SemVoxelCloud.getNumberOfGrids(); i++) {
//                PointT p_sem;
//                SemanticVoxelGrid vg = SemVoxelCloud.getGridList()[i];
//                Eigen::Vector3f gc = vg.getCentroid(SemVoxelCloud.getLeafSize());
//                p_sem.x = gc[0];
//                p_sem.y = gc[1];
//                p_sem.z = gc[2];
//
//                /*Eigen::Vector3i bgr = getColorVector(vg.getSemanticVector());
//                p_sem.b = bgr[0];
//                p_sem.g = bgr[1];
//                p_sem.r = bgr[2];*/
//
//                int class_index = argmax(vg.getSemanticVector());
//                p_sem.b = cmap_data[class_index];
//                p_sem.g = cmap_data[class_index + 1];
//                p_sem.r = cmap_data[class_index + 2];
//
//                if (p_sem.z != 0)
//                    SemPointCloudFiltered->points.push_back(p_sem);
//            }
//            std::chrono::steady_clock::time_point t10 = std::chrono::steady_clock::now();
//            cout << "Semantic Point Cloud Filtering Time: "
//                 << std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count()
//                 << "us."  << endl;
//
//            std::cout << "SemPointCloud after filtering has: " << SemPointCloud->points.size() << " data points."
//                      << std::endl;
//            pcl::io::savePCDFileBinary(FLAGS_output_path + FLAGS_net_name + ".semantic.filtered.pcd", *SemPointCloud);

        }

        // Py_Finalize();
        return 0;
    }
}

/*
 * argmax: Return the index of the element which has the maximum probability in the semantic vector
 */
int argmax(float* vector)
{
    float max = vector[0];
    int argmax = 0;
    for(int i = 1; i < NUM_OF_CLASSES; i++)
    {
        if(vector[i] > max)
        {
            max = vector[i];
            argmax = i;
        }
    }
    return argmax;
}