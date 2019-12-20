#include <iostream>
#include "NeuralNetwork.h"

int main() {


    for (int i = 0; i < 10; i++) {
        Layer l(5);
        std::cout << l.getID() << std::endl;
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
