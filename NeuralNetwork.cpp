//
// Created by dgageot on 20/12/2019.
//

#include <stdexcept>
#include "NeuralNetwork.h"

void NeuralNetwork::feed(const std::vector<double> &inputs) {
    // Feed layers in sequence
    for (auto it = _layers.begin(); it != _layers.end(); ++it) {
        // If it is the first layer, feed it with input data
        if (it == _layers.begin()) {
            it->feed(inputs);
        } else {
            // Otherwise feed it with the precedent layer
            it->feed(*(it - 1));
        }
    }
}

std::vector<double> NeuralNetwork::getOutputs() const {
    if (_layers.empty()) {
        throw std::runtime_error("No layers in the neural network");
    }
    return _layers.back().getOutputs();
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &sizes) {
    for (auto size : sizes) {
        _layers.emplace_back(Layer(size));
    }
}

void NeuralNetwork::train(const std::pair<std::vector<double>, std::vector<double>> &values) {

    // Feed the neural network with the input

    // Compute the cost and error

    // Back propagate

    // Readjust weights and biases


}
