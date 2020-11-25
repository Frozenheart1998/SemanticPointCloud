//
// Created by lincoln on 11/15/20.
// Mfied from Point Cloud Library
// Fl/filters/voxel_grid.h
//

#ifndef PROJECTION_SEMANTICVOXELCLOUD_H
#define PROJECTION_SEMANTICVOXELCLOUD_H

#include <vector>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include "SemanticVoxelGrid.h"
#include "ProcessVector.h"

class SemanticVoxelCloud {
public:
    SemanticVoxelCloud()
    {
        Eigen::Vector3f leaf_size(0.1f, 0.1f, 0.1f);
        SemanticVoxelCloud(leaf_size, 0);
    }

    SemanticVoxelCloud(Eigen::Vector3f leaf_size, int numberOfGrids)
    {
        this->leaf_size = leaf_size;
        this->numberOfGrids = numberOfGrids;
    }

    ~SemanticVoxelCloud(){}

    inline void addPoint(float x, float y, float z, float* semanticVector)
    {
        Eigen::Vector3i gridCoordinate = getGridCoordinates(x, y, z);
        SemanticVoxelGrid *current;
        if((current = getSemanticVoxelGrid(gridCoordinate)) == NULL)
        {
            current = new SemanticVoxelGrid(gridCoordinate);
            numberOfGrids++;
        }
        current->addSemanticVector(semanticVector);
        gridList.push_back(*current);
    }

    inline void addPoint(float x, float y, float z, float* semanticVector, size_t index)
    {
        Eigen::Vector3i gridCoordinate = getGridCoordinates(x, y, z);
        SemanticVoxelGrid *current;
        if(index == 0)
        {
            current = new SemanticVoxelGrid(gridCoordinate);
            numberOfGrids++;
            return;
        }
        current->addSemanticVector(semanticVector);
        gridList.push_back(*current);
    }

    /** \brief Returns the corresponding (i,j,k) coordinates in the grid of point (x,y,z).
      * \param[in] x the X point coordinate to get the (i, j, k) index at
      * \param[in] y the Y point coordinate to get the (i, j, k) index at
      * \param[in] z the Z point coordinate to get the (i, j, k) index at
      */
    inline Eigen::Vector3i
    getGridCoordinates (float x, float y, float z) const
    {
        return (Eigen::Vector3i (static_cast<int> (std::floor (x / leaf_size[0])),
                                 static_cast<int> (std::floor (y / leaf_size[1])),
                                 static_cast<int> (std::floor (z / leaf_size[2]))));
    }

    SemanticVoxelGrid* getSemanticVoxelGrid(Eigen::Vector3i des);
    SemanticVoxelGrid* getSemanticVoxelGrid(Eigen::Vector4i des);

    inline void setLeafSize(Eigen::Vector3f leaf_size) { this->leaf_size = leaf_size; }
    inline Eigen::Vector3f getLeafSize() { return leaf_size; }
    inline void setNumberOfGrids(int numberOfGrids) { this->numberOfGrids = numberOfGrids; }
    inline int getNumberOfGrids() { return numberOfGrids; }
    inline std::vector<SemanticVoxelGrid> getGridList() { return gridList; }

private:
    Eigen::Vector3f leaf_size;
    int numberOfGrids;
    std::vector<SemanticVoxelGrid> gridList;
};


#endif //PROJECTION_SEMANTICVOXELCLOUD_H
