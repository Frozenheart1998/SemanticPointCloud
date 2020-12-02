//
// Created by lincoln on 11/15/20.
//

#include "Python.h"
#include<iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Geometry>

#include "cnpy.h"

using namespace std;

int main()
{
    float vector[40] = {0};
    int bgr[3] = {0};
    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\'/home/lincoln/CLionProjects/Projection\')");
//    PyRun_SimpleString("import numpy as np");
//    PyRun_SimpleString("a = np.load(\'/media/lincoln/Lincoln/downloads/semantic.npy/frame-000000.npy\')[0][0]");
//    PyRun_SimpleString("b = np.load(\'/media/lincoln/Lincoln/downloads/semantic.npy/frame-000000.npy\')[0][0]");
//    PyRun_SimpleString("print(a,b,a+b)");

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
    PyObject* arg0 = PyUnicode_FromString("/media/lincoln/Lincoln/downloads/semantic.npy/frame-000000.npy");    // argument 0
    PyObject* arg1 = PyLong_FromLong(0);    // arg 1
    PyObject* arg2 = PyLong_FromLong(0);    // arg 2
    PyTuple_SetItem(args, 0, arg0);
    PyTuple_SetItem(args, 1, arg1);
    PyTuple_SetItem(args, 2, arg2);

    PyObject* pReturn = PyObject_CallObject(pFunc, args);

    if(PyList_Check(pReturn)){ //Check whether is Python List or not
        // printf("Return Result： ");
        int SizeOfList = PyList_Size(pReturn);// The Size of List object
        for(int i = 0; i < SizeOfList; i++){
            PyObject *Item = PyList_GetItem(pReturn, i);// Fetch each element in the list
            PyArg_Parse(Item, "f", &vector[i]);// "f" denotes float
//            printf("%f ",result);
        }
//        printf("\n");
    }else{
        cout<<"Not a List"<<endl;
    }

    PyObject *pFunc2 = PyObject_GetAttrString(pModule, "SemToRGB");
    if (pFunc2 == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to PyObject_GetAttrString");
    }

    PyObject* PyList  = PyList_New(40);//定义一个与数组等长的PyList对象数组
    PyObject* args2 = PyTuple_New(1);//定义一个Tuple对象，Tuple对象的长度与Python函数参数个数一直，上面Python参数个数为1，所以这里给的长度为1
    for(int i = 0; i < PyList_Size(PyList); i++)
        PyList_SetItem(PyList,i, PyFloat_FromDouble(vector[i]));//给PyList对象的每个元素赋值
    PyTuple_SetItem(args2, 0, PyList);//将PyList对象放入PyTuple对象中

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    PyObject* pReturn2 = PyObject_CallObject(pFunc2, args2);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    cout << "SemToRGB Time: "
         << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()
         << "us.\n";

    if(PyList_Check(pReturn2)){ //Check whether is Python List or not
        // printf("Return Result： ");
        int SizeOfList = PyList_Size(pReturn2);// The Size of List object
        for(int i = 0; i < SizeOfList; i++){
            PyObject *Item = PyList_GetItem(pReturn2, i);// Fetch each element in the list
            int result;
            PyArg_Parse(Item, "i", &bgr[i]);// "f" denotes float
            printf("%d ",bgr[i]);
        }
        printf("\n");
    }else{
        cout<<"Not a List"<<endl;
    }



    Py_Finalize();

//    for(int i = 0; i < 40; i++)
//    {
//        cout << vector[i] << endl;
//    }

/*    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cnpy::NpyArray npydata = cnpy::npy_load("/media/lincoln/Lincoln/downloads/semantic.npy/frame-000000.npy");
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    cout << "Time: "
         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
         << "us.\n";
    float* data = npydata.data<float>();
    cout << npydata.num_bytes() / sizeof(float) / 40 << " " << data[0] << " " << data[1] << endl;

    cnpy::NpyArray npydata2 = cnpy::npy_load("../cmap.npy");
    uint8_t * cmap = npydata.data<uint8_t>();

    cout << (int)cmap[0] << " " << (int)cmap[1] << " " << (int)cmap[2] << endl;*/

    ifstream fin("/home/lincoln/Documents/dataset/out-recording49.sens/frame-000721.pose.txt");
    if (!fin) {
        cerr << "Pose txt excluded!" << endl;
        return 1;
    }

    double data[12] = {0};
    for (auto &d:data) {
        fin >> d;
    }
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
        cout << "False Pose" << endl;
    }
    Eigen::Matrix3d rotMatrix;
    rotMatrix << data[0], data[1], data[2],
            data[4], data[5], data[6],
            data[8], data[9], data[10];
    Eigen::Isometry3d T(rotMatrix);
    T.pretranslate(Eigen::Vector3d(data[3], data[7], data[11]));

    return 0;
}
