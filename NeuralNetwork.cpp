//
// Created by dgageot on 20/12/2019.
//

#include <stdexcept>
#include <algorithm>
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

void NeuralNetwork::train(const std::pair<std::vector<double>, std::vector<double>> &stimuli) {

    // Retrieve reference to input and output data
    const std::vector<double>& inputs = stimuli.first;
    const std::vector<double>& outputs = stimuli.second;

    // Feed the neural network with the input
    feed(inputs);

    // Backpropagate
    backpropagate(outputs);

    // Readjust weights and biases
    gradientDescent(inputs);

}

void NeuralNetwork::backpropagate(const std::vector<double> &expectedOutputs) {
    // Compute cost error
    std::vector<double> errorOutputs = getOutputs();
    std::transform(errorOutputs.begin(), errorOutputs.end(), expectedOutputs.begin(),
                   errorOutputs.begin(), std::minus<>());

    for (auto it = _layers.end() - 1; it != _layers.begin() - 1; --it) {
        // If it is the first layer, feed it with input data
        if (it == _layers.end() - 1) {
            it->backpropagate(errorOutputs);
        } else {
            // Otherwise feed it with the precedent layer
            it->backpropagate(*(it+1));
        }
    }
}

void NeuralNetwork::gradientDescent(const std::vector<double> &inputs) {
    for (auto it = _layers.end() - 1; it != _layers.begin() - 1; --it) {
        // If it is the first layer, feed it with input data
        if (it == _layers.begin()) {
            it->gradientDescent(inputs, 0);
        } else {
            // Otherwise feed it with the precedent layer
            it->gradientDescent(*(it-1), 0);
        }
    }
}
