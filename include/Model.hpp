#pragma once

#include <vector>
#include <string>

typedef unsigned int uint;

class Model {
    public:
    // constructor
    Model(uint numLayers, uint neuronsPerLayer, float learningRate = 0.05);

    // weight vector to string for printing to console
    std::string weightsToString();

    // attributes
    std::vector<std::vector<uint>*> layers; // vector matrix
    uint learningRate;
    std::vector<std::vector<std::vector<float>*>*> weights;
};