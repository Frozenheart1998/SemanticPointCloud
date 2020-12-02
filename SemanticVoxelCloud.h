//
// Created by lincoln on 11/15/20.
// Mfied from Point Cloud Library
// Fl/filters/voxel_grid.h
//

#ifndef PROJECTION_SEMANTICVOXELCLOUD_H
#define PROJECTION_SEMANTICVOXELCLOUD_H

#include <vector>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include "SemanticVoxelGrid.h"
#include "ProcessVector.h"
#include "cnpy.h"

#define CMAP_ROW 256
#define CMAP_COL 3

// defines pcl type
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef Eigen::Matrix<float, 40, 1> Vector40f;

class SemanticVoxelCloud {
public:
    SemanticVoxelCloud(PointCloud::Ptr& SemPointCloud, PointCloud::Ptr& SemPointCloudFiltered)
    {
        this->SemPointCloud = SemPointCloud;
        this->SemPointCloudFiltered = SemPointCloudFiltered;
        start_frame = 0;
        frame_num = 1;
        steps = 1;
    }

    ~SemanticVoxelCloud(){}

    void filter()
    {
        filter(0.1, 0.1, 0.1, true);
    }

    void filter(float lx, float ly, float lz, bool save_leaf_layout)
    {
        PointCloud::Ptr temp(new PointCloud);
        voxel_grid.setInputCloud(SemPointCloud);
        voxel_grid.setLeafSize(lx, ly, lz);//change leaf size into 0.5cm
        voxel_grid.setSaveLeafLayout(save_leaf_layout);
        voxel_grid.filter(*temp);

        gridList.resize(voxel_grid.getLeafLayout().size());
        for (int i = 0; i < gridList.size(); ++i)
        {
            gridList[i].setXYZ(Eigen::Vector3f::Zero());
            gridList[i].setSemanticVector(Vector40f::Zero());
            gridList[i].setNumberOfPoints(0);
        }

        // Camera Calibration
        double fx = 366.41;
        double fy = 366.41;
        double cx = 251.454;
        double cy = 205.593;
        double depthScale = 1000.0;

        cv::Mat depthImg;
        // Semantic Image
        cnpy::NpyArray semanImg;
        for (int frame = start_frame; frame < frame_num; frame += steps)
        {
            if (frame % 100 == 0)
                cout << "Semantic Filter: " << frame << endl;
            char frame_string[12];
            sprintf(frame_string, "frame-%06d", frame);

            ifstream fin(input_path + frame_string + ".pose.txt");
            if (!fin) {
                cerr << "Pose txt excluded!" << endl;
                exit(1);
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

            depthImg = cv::imread(input_path + frame_string + ".depth.pgm", -1);
            // Semantic Vector Loading
            semanImg = cnpy::npy_load(sem_input_path + frame_string + ".npy");

            for (int v = 0; v < depthImg.rows; v++)
                for (int u = 0; u < depthImg.cols; u++) {
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    ushort d = depthImg.ptr<ushort>(v)[u];
                    if (d == 0)
                        continue;
                    Eigen::Vector3d point;
                    point[2] = static_cast<float>(d) / depthScale; // depth
                    point[0] = (u - cx) * point[2] / fx;
                    point[1] = (v - cy) * point[2] / fy;
                    Eigen::Vector3d pointWorld = T * point;
                    pointWorld[1] = -pointWorld[1];
                    pointWorld[2] = -pointWorld[2];

                    Vector40f semanticVector;
                    for (int idx = 0; idx < NUM_OF_CLASSES; idx++) {
                        semanticVector(idx) = *(semanImg.data<float>() + (depthImg.cols * v + u) * NUM_OF_CLASSES +
                                                idx);
                    }

                    PointT p;
                    p.x = pointWorld[0];
                    p.y = pointWorld[1];
                    p.z = pointWorld[2];

                    int centroid_index = voxel_grid.getCentroidIndex(p);
                    gridList[centroid_index].addPoint(pointWorld, semanticVector);
                }
        }

        cnpy::NpyArray cmap = cnpy::npy_load("../cmap.npy");
        uint8_t *cmap_data = cmap.data<uint8_t>();
        for (int i = 0; i < gridList.size(); i++)
        {
            PointT p;
            Eigen::Vector3f xyz = gridList[i].getXYZ();
            Vector40f vector = gridList[i].getSemanticVector();
            int n = gridList[i].getNumberOfPoints();
            float inv_n = 1 / static_cast<float>(n);
            p.x = xyz[0] * inv_n;
            p.y = xyz[1] * inv_n;
            p.z = xyz[2] * inv_n;

            int class_index = argmax(vector);
            p.b = cmap_data[class_index + 2 * CMAP_ROW];
            p.g = cmap_data[class_index + CMAP_ROW];
            p.r = cmap_data[class_index];
            SemPointCloudFiltered->points.push_back(p);

        }
    }

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

    inline void setSemPointCloud(PointCloud::Ptr& SemPointCloud) { this->SemPointCloud = SemPointCloud; }
    inline PointCloud::Ptr getSemPointCloud() { return SemPointCloud; }
    inline void setSemPointCloudFiltered(PointCloud::Ptr& SemPointCloudFiltered) { this->SemPointCloudFiltered = SemPointCloudFiltered; }
    inline PointCloud::Ptr getSemPointCloudFiltered() { return SemPointCloudFiltered; }
    inline void setVoxelGrid(pcl::VoxelGrid<pcl::PointXYZRGB>& voxel_grid) { this->voxel_grid = voxel_grid; }
    inline pcl::VoxelGrid<pcl::PointXYZRGB> getVoxelGrid() { return voxel_grid; }
    inline std::vector<SemanticVoxelGrid> getGridList() { return gridList; }

    inline void setInputPath(string input_path) { this->input_path = input_path; }
    inline void setSemInputPath(string sem_input_path) { this->sem_input_path = sem_input_path; }
    inline void setStartFrame(int start_frame) { this->start_frame = start_frame; }
    inline void setFrameNum(int frame_num) { this->frame_num = frame_num; }
    inline void setSteps(int steps) { this->steps = steps; }

private:
    PointCloud::Ptr SemPointCloud, SemPointCloudFiltered;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    std::vector<SemanticVoxelGrid> gridList;
    string input_path, sem_input_path;
    int start_frame, frame_num, steps;
};


#endif //PROJECTION_SEMANTICVOXELCLOUD_H
