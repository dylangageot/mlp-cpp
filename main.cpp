#include <iostream>
#include "NeuralNetwork.h"

int main() {

    NeuralNetwork network({2,1});

    for (int i = 0; i < 900000; ++i) {
        network.train({{1,1}, {1}});
        network.train({{0,1}, {0}});
        network.train({{1,0}, {0}});
        network.train({{0,0}, {0}});
    }


    network.feed({0,1});

    for (auto elt : network.getOutputs()) {
        std::cout << elt << " ";
    }

    return 0;
}
