#pragma once

#include <vector>
#include <string>

typedef unsigned int uint;

class Model {
    private:
    // attributes 
    std::vector<std::vector<uint>*> layers; // holds nodes
    std::vector<std::vector<float>> biases; // holds biases for nodes, except input 
    uint learningRate;
    std::vector<std::vector<std::vector<float>*>*> weights;

    // initializes biases
    void init_biases();

    // used for changing float to string
    template <typename T> std::string toStr(const T& t);

 
    public:
    // constructor
    Model(uint numLayers, uint neuronsPerLayer, float learningRate = 0.05);

    // weight vector to string for printing to console
    std::string weightsToString();

    // layer vector to string for printing to console
    std::string layersToString();
};