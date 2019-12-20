#include <iostream>
#include "NeuralNetwork.h"

int main() {

    NeuralNetwork network({2,1});
    network.feed({1,1});

    for (auto elt : network.getOutputs()) {
        std::cout << elt << " ";
    }

    return 0;
}
