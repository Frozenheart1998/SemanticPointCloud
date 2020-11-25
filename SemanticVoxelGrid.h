//
// Created by lincoln on 11/15/20.
//

#ifndef PROJECTION_SEMANTICVOXELGRID_H
#define PROJECTION_SEMANTICVOXELGRID_H

#include <vector>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include "ProcessVector.h"
//#include "NumCpp.hpp"

#define NUM_OF_CLASSES 40

class SemanticVoxelGrid {
public:
    SemanticVoxelGrid()
    {
        SemanticVoxelGrid(0, 0, 0);
    }

    SemanticVoxelGrid(Eigen::Vector4i gridCoordinate)
    {
        SemanticVoxelGrid(gridCoordinate[0], gridCoordinate[1], gridCoordinate[2]);
    }

    SemanticVoxelGrid(Eigen::Vector3i gridCoordinate)
    {
        SemanticVoxelGrid(gridCoordinate[0], gridCoordinate[1], gridCoordinate[2]);
    }

    SemanticVoxelGrid(int x0, int x1, int x2)
    {
        this->gridCoordinate << x1, x2, x2;
        memset(semanticVector, 0, sizeof(float) * NUM_OF_CLASSES);
        numberOfPoints = 0;
//        bgr << 0, 0, 0;
    }


    ~SemanticVoxelGrid(){}

    void addSemanticVector (float* semanticVector)
    {
        for(int i = 0; i < NUM_OF_CLASSES; i++)
        {
            this->semanticVector[i] += semanticVector[i];
        }
        numberOfPoints++;
    }

    // inline void setGridColor() { bgr = getColorVector(semanticVector); }

    inline Eigen::Vector3i getGridCoordinate() { return gridCoordinate; }
    inline float* getSemanticVector() { return semanticVector; }
    inline int getNumberOfPoints() { return numberOfPoints; }
    inline Eigen::Vector3f getCentroid(Eigen::Vector3f size)
    {
        return Eigen::Vector3f(size[0] * ((float)gridCoordinate[0] + 0.5),
                               size[1] * ((float)gridCoordinate[1] + 0.5),
                               size[2] * ((float)gridCoordinate[2] + 0.5));
    }

    // inline Eigen::Vector3i getBGR() { return bgr; }

private:
    Eigen::Vector3i gridCoordinate;
    float semanticVector[40];
    int numberOfPoints;
    // Eigen::Vector3i bgr;
};


#endif //PROJECTION_SEMANTICVOXELGRID_H
