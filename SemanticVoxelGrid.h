//
// Created by lincoln on 11/15/20.
//

#ifndef PROJECTION_SEMANTICVOXELGRID_H
#define PROJECTION_SEMANTICVOXELGRID_H

#include <vector>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>

typedef Eigen::Matrix<float, 40, 1> Vector40f;

#define NUM_OF_CLASSES 40
#define NUM_OF_SMALL_CLASSES 13

class SemanticVoxelGrid {
public:
    SemanticVoxelGrid()
    {
        SemanticVoxelGrid(0, 0, 0);
    }

    SemanticVoxelGrid(Eigen::Vector4f gridCoordinate)
    {
        SemanticVoxelGrid(gridCoordinate[0], gridCoordinate[1], gridCoordinate[2]);
    }

    SemanticVoxelGrid(Eigen::Vector3f gridCoordinate)
    {
        SemanticVoxelGrid(gridCoordinate[0], gridCoordinate[1], gridCoordinate[2]);
    }

    SemanticVoxelGrid(float x0, float x1, float x2)
    {
        this->xyz<< x1, x2, x2;
        for (int i = 0; i < semanticVector.size(); i++)
        {
            semanticVector[0] = 0;
        }
        numberOfPoints = 0;
    }


    ~SemanticVoxelGrid(){}

    void addPoint (Eigen::Vector3d xyz, Vector40f semanticVector)
    {
        this->xyz[0] += xyz[0];
        this->xyz[1] += xyz[1];
        this->xyz[2] += xyz[2];

        for(int i = 0; i < NUM_OF_CLASSES; i++)
        {
            this->semanticVector(i) += semanticVector(i);
        }

        numberOfPoints++;
    }

    inline Eigen::Vector3f getXYZ() { return xyz; }
    inline Vector40f getSemanticVector() { return semanticVector; }
    inline int getNumberOfPoints() { return numberOfPoints; }

    inline void setNumberOfPoints(int numberOfPoints) { this->numberOfPoints = numberOfPoints; }
    inline void setXYZ(Eigen::Vector3f xyz) { this->xyz[0] = xyz[0]; this->xyz[1] = xyz[1]; this->xyz[2] = xyz[2]; }
    void setSemanticVector(Vector40f semanticVector)
    {
        for(int i = 0; i < NUM_OF_CLASSES; i++)
        {
            this->semanticVector(i) = semanticVector(i);
        }
    }


    // inline Eigen::Vector3i getBGR() { return bgr; }

private:
    Eigen::Vector3f xyz;
    Vector40f semanticVector;
    int numberOfPoints;
};


#endif //PROJECTION_SEMANTICVOXELGRID_H
