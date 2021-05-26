#pragma once

#include <vector>

typedef unsigned int uint;

class Model {
    public:
    // constructor
    Model(uint numlayers, uint neuronsPerLayer, float learningRate = 0.05);

    // attributes
    std::vector<std::vector<uint>*> layers; // vector matrix
    uint learningRate;
    std::vector<int> weights;
};