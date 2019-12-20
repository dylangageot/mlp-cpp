//
// Created by dgageot on 20/12/2019.
//

#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include "Layer.h"

#define MLP_VERBOSE

int Layer::_id_counter = 0;


void Layer::feed(const Layer &layer) {

}

void Layer::setUp(const Layer &input) {

    // Check sizes of weights vector
    int expectedSize = input.getSize() * this->getSize();
    if (expectedSize != _weigths.size()) {
#ifdef MLP_VERBOSE
        std::cout << "Build weigths vector for layer " << this->getID() << " of size " << expectedSize << std::endl;
#endif
        _weigths.resize(expectedSize);
        static std::uniform_real_distribution<double> distribution(0.0f, 1.0f); //Values between 0 and 2
        static std::mt19937 engine; // Mersenne twister MT19937
        static auto generator = std::bind(distribution, engine);
        std::generate_n(_weigths.begin(), expectedSize, generator);
    }

}
