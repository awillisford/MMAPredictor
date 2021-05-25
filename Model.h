#pragma once

typedef unsigned int uint;

class Model {
    public:
    // constructor
    Model(uint layers, uint neuronsPerLayer, float learningRate = 0.05);

    // attributes
    uint layers;
    uint neuronsPerLayer;
    uint learningRate;
};