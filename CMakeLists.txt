cmake_minimum_required(VERSION 3.14)
project(mlp)

set(CMAKE_CXX_STANDARD 14)

add_executable(mlp main.cpp NeuralNetwork.cpp NeuralNetwork.h Layer.cpp Layer.h MathFunctions.cpp MathFunctions.h)