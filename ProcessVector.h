//
// Created by lincoln on 11/15/20.
//
#ifndef PROJECTION_PROCESSVECTOR_H
#define PROJECTION_PROCESSVECTOR_H

#include <Eigen/Geometry>

#define NUM_OF_CLASSES 40

using namespace std;

void getSemanticVector(char* filepath, int row, int col, float* vector);
Eigen::Vector3i getColorVector(float* vector);

#endif