#pragma once

#include <vector>

typedef unsigned int uint;

class Model {
    public:
    // constructor
    Model(std::vector<uint> layers, uint neuronsPerLayer, float learningRate = 0.05);

    // attributes
    uint layers;
    uint learningRate;
    std::vector<uint> weights;
};