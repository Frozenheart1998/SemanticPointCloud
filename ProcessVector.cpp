//
// Created by lincoln on 11/17/20.
//

#include "Python.h"
#include <iostream>
#include <string>
#include <Eigen/Geometry>

#define NUM_OF_CLASSES 40

using namespace std;

/*void getSemanticVector(char* filepath, int row, int col, float* vector);
Eigen::Vector3i getColorVector(float* vector);*/

void getSemanticVector(char* filepath, int row, int col, float* vector)
{
//    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\'/home/lincoln/CLionProjects/Projection\')");

    PyObject *pModule = PyImport_ImportModule("loadSemanticVector");
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "loadSemanticVector");
    if (pFunc == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to PyObject_GetAttrString");
    }

    PyObject* args = PyTuple_New(3);   // 3 Agumrnts
    PyObject* arg0 = PyString_FromString(filepath);    // argument 0
//    PyObject* arg0 = PyString_FromString("/media/lincoln/Lincoln/downloads/semantic.npy/frame-000000.npy");
    PyObject* arg1 = PyLong_FromLong(row);    // arg 1
    PyObject* arg2 = PyLong_FromLong(col);    // arg 2
    PyTuple_SetItem(args, 0, arg0);
    PyTuple_SetItem(args, 1, arg1);
    PyTuple_SetItem(args, 2, arg2);

    PyObject* pReturn = PyObject_CallObject(pFunc, args);

    if(PyList_Check(pReturn)){ //Check whether is Python List or not
        int SizeOfList = PyList_Size(pReturn);// The Size of List object
        for(int i = 0; i < SizeOfList; i++){
            PyObject *Item = PyList_GetItem(pReturn, i);// Fetch each element in the list
            PyArg_Parse(Item, "f", &vector[i]);// "f" denotes float
        }
    }else{
        cout<<"Not a List"<<endl;
    }

//    Py_Finalize();
}

Eigen::Vector3i getColorVector(float* vector)
{
    int bgr_array[3];
    Eigen::Vector3i bgr;
//    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\'/home/lincoln/CLionProjects/Projection\')");

    PyObject *pModule = PyImport_ImportModule("loadSemanticVector");
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "SemToRGB");
    if (pFunc == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to PyObject_GetAttrString");
    }

    PyObject* PyList  = PyList_New(40);//定义一个与数组等长的PyList对象数组
    PyObject* args = PyTuple_New(1);//定义一个Tuple对象，Tuple对象的长度与Python函数参数个数一直，上面Python参数个数为1，所以这里给的长度为1
    for(int i = 0; i < PyList_Size(PyList); i++)
        PyList_SetItem(PyList,i, PyFloat_FromDouble(vector[i]));//给PyList对象的每个元素赋值
    PyTuple_SetItem(args, 0, PyList);//将PyList对象放入PyTuple对象中

    PyObject* pReturn = PyObject_CallObject(pFunc, args);

    if(PyList_Check(pReturn)){ //Check whether is Python List or not
        int SizeOfList = PyList_Size(pReturn);// The Size of List object
        for(int i = 0; i < SizeOfList; i++){
            PyObject *Item = PyList_GetItem(pReturn, i);// Fetch each element in the list
            int result;
            PyArg_Parse(Item, "i", &bgr_array[i]);// "f" denotes float
        }
    }else{
        cout<<"Not a List"<<endl;
    }

//    Py_Finalize();

    bgr << bgr_array[0], bgr_array[1], bgr_array[2];

    return bgr;
}
