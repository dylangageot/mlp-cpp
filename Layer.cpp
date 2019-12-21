//
// Created by dgageot on 20/12/2019.
//

#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include "Layer.h"
#include "MathFunctions.h"

#define MLP_VERBOSE

int Layer::_id_counter = 0;

void Layer::feed(const Layer &input) {
    // Feed with output of previous layer
    feed(input.getOutputs());
}

void Layer::feed(const std::vector<double>& input) {
    // Set-up the layer
    setUp(input);

    // Compute output values
    for (int i = 0; i < _size; ++i) {
        double sigmoid_input = std::inner_product(std::begin(_weights.at(i)),std::end(_weights.at(i)),
                                                  std::begin(input), _biases.at(i));
        _weightedInputs.at(i) = sigmoid_input;
        _outputs.at(i) = sigmoid(sigmoid_input);
    }
}

void Layer::setUp(const std::vector<double> &input) {
    // If left layer output sizes does not correspond, initialize weights
    const int expectedSize = input.size();
    if (expectedSize != _weights.at(0).size()) {

        static std::uniform_real_distribution<double> distribution(0.0f, 1.0f); //Values between 0 and 1
        static std::default_random_engine engine;

        // Resize and generate random value for weights
        for (auto& weights : _weights) {
            weights.resize(expectedSize);
            std::generate(weights.begin(), weights.end(), [](){return distribution(engine);});
        }
        // Generate random value for biases
        std::generate(_biases.begin(), _biases.end(), [](){return distribution(engine);});


#ifdef MLP_VERBOSE
        std::cout << "Build weigths vector for layer " << this->getID() << " of size " << expectedSize * _size
                  << std::endl;
#endif
    }
}

std::vector<double> Layer::getWeightedErrors() const {

    std::vector<double> weightedErrors(_weights.at(0).size());

    for (auto it = weightedErrors.begin(); it != weightedErrors.end(); ++it) {
        std::vector<double> transposedVector(_size);
        int index = it - weightedErrors.begin();
        // Transpose computation
        std::transform(_weights.begin(), _weights.end(),
                       transposedVector.begin(), [index](std::vector<double> in) { return in.at(index); });
        // Inner product computation
        *it = std::inner_product(transposedVector.begin(),transposedVector.end(), _gradientErrors.begin(), 0.0);
    }

    return weightedErrors;
}

void Layer::backpropagate(const std::vector<double> &weightedErrors) {
    // Compute errors on the layer
    std::transform(weightedErrors.begin(), weightedErrors.end(), _weightedInputs.begin(), _gradientErrors.begin(),
            [](double e, double z) { return e - e*sigmoid_derived(z);});
}

void Layer::backpropagate(const Layer &layer) {
    backpropagate(layer.getWeightedErrors());
}


void Layer::gradientDescent(const std::vector<double> &inputs, int miniBatchSize) {
    // Adjust weights and biases

    // 1. Adjust weights
    for (auto it1 = _weights.begin(); it1 != _weights.end(); ++it1) {
        int index1 = it1 - _weights.begin();
        for (auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
            int index2 = it2 - it1->begin();
            *(it2) -= LEARNING_RATE * inputs.at(index2) * _gradientErrors.at(index1);
        }
    }

    // 2. Adjust biases
    std::transform(_biases.begin(), _biases.end(), _gradientErrors.begin(), _biases.begin(),
            [](double b, double e) { return b - LEARNING_RATE * e; });

}

void Layer::gradientDescent(const Layer &input, int miniBatchSize) {
    gradientDescent(input.getOutputs(), miniBatchSize);
}

void Layer::resetGradientErrors() {
    _gradientErrors.clear();
}

