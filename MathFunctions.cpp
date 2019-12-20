//
// Created by Dylan-PC on 20/12/2019.
//

#include "MathFunctions.h"

double sigmoid(double z) {
    return 1 / ( 1 + exp(-z) );
}

double sigmoid_derived(double z) {
    return z*exp(-z) / ((1+exp(-z))*(1+exp(-z)));
}