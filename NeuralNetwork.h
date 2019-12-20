//
// Created by dgageot on 20/12/2019.
//

#ifndef MLP_NEURALNETWORK_H
#define MLP_NEURALNETWORK_H


#include <vector>
#include "Layer.h"

class NeuralNetwork {

public:
    NeuralNetwork() = default;
    void feedForward(const std::vector<double>& inputs);
//    std::vector<double>& results() const;

protected:
    std::vector<Layer> layers;

};


#endif //MLP_NEURALNETWORK_H
