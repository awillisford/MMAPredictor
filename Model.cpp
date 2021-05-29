#include "Model.h"
#include <iostream>

Model::Model(std::vector<float> input, uint numLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create layers
    weights.push_back(new std::vector<std::vector<float>*>);
    for (uint x = 0; x < numLayers; ++x) {
        layers.push_back(new std::vector<uint>);
        // create neurons
        for (uint z = 0; z < neuronsPerLayer; ++z) {
            /* I am using the arrow operator in this situation
             * since each element in layers is a vector pointer */
            layers[x]->push_back(z); 
        }
    }
}

int main() {
    std::cout << "Working (currently)...";
    return 0;
}