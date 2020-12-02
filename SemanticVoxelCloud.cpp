//
// Created by lincoln on 11/15/20.
//

#include "SemanticVoxelCloud.h"
#include "SemanticVoxelGrid.h"

/*
SemanticVoxelGrid* SemanticVoxelCloud::getSemanticVoxelGrid(Eigen::Vector3i des)
{
    for(int i = numberOfGrids - 1; i >= 0; i--)
    {
        Eigen::Vector3i src = gridList[i].getGridCoordinate();
        if(src == des) return (&gridList[i]);
    }
    return NULL;
}

SemanticVoxelGrid* SemanticVoxelCloud::getSemanticVoxelGrid(Eigen::Vector4i des)
{
    Eigen::Vector3i newDes(des[0], des[1], des[2]);
    getSemanticVoxelGrid(newDes);
}*/
