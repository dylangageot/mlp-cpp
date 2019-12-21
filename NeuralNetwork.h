//
// Created by dgageot on 20/12/2019.
//

#ifndef MLP_NEURALNETWORK_H
#define MLP_NEURALNETWORK_H


#include <vector>
#include "Layer.h"

class NeuralNetwork {

public:
    /**
     * Construct a neural network with a specified topology
     * @param sizes vector containing size for the layers
     */
    explicit NeuralNetwork(const std::vector<int>& sizes);
    /**
     * Feed the neural network with input data
     * @param inputs input data
     */
    void feed(const std::vector<double>& inputs);
    /**
     * Get output data of the neural network
     * @return output data
     */
    std::vector<double> getOutputs() const;

    void train(const std::pair<std::vector<double>, std::vector<double>>& values);


protected:
    /**
     * Ordered vector of layers
     */
    std::vector<Layer> _layers;


    void backpropagate(const std::vector<double>& expectedOutputs);

    void gradientDescent(const std::vector<double>& inputs);

};

#endif //MLP_NEURALNETWORK_H
