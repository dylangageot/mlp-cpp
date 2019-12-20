//
// Created by Dylan-PC on 20/12/2019.
//

#ifndef MLP_MATHFUNCTIONS_H
#define MLP_MATHFUNCTIONS_H

#include <math.h>

/**
 * Sigmoid function
 * @param z input parameter
 * @return output of sigmoid function of z
 */
double sigmoid(double z);

/**
 * Derived sigmoid function
 * @param z input parameter
 * @return output of derived sigmoid function of z
 */
double sigmoid_derived(double z);

#endif //MLP_MATHFUNCTIONS_H
