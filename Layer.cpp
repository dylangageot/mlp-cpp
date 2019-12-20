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