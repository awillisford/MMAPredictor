#include "Model.h"
#include <iostream>

Model::Model(std::vector<std::vector<float>> input, uint numLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create vector for input layer's weights
    weights.push_back(new std::vector<std::vector<float>*>);
    // create a weight vector that holds type float for each node in input
    for (int it = 0; it < input[0].size() * 2; ++it) { // 2 inputs
        weights[0]->push_back(new std::vector<float>);
    }

    // create layers ----- UPDATE COMMENT
    for (uint x = 0; x < numLayers; ++x) {
        layers.push_back(new std::vector<uint>);
        weights.push_back(new std::vector<std::vector<float>*>); // create hidden layer weights
        // create neurons  ------- UPDATE COMMENT
        for (uint z = 0; z < neuronsPerLayer; ++z) {
            layers[x]->push_back(z);
            weights[x + 1]->push_back(new std::vector<float>); // creates weight vector for each node in hidden layers
        }
    }

    // assign weights for each node past input layer
    for (uint x = 0; x < numLayers + 1; ++x) {
        float temp = 1.0; // temporary float variable to push back into vector
        for (uint z = 0; z < neuronsPerLayer; ++z) {
            // assign weights per node based on number of neurons in each layer
            if (x < numLayers) {
                for (int it = 0; it < neuronsPerLayer; ++it) {
                    (*weights[x])[z]->push_back(temp);
                }
            }
            // create two weights per node in last hidden layer since there are two outputs
            else {
                for (int it = 0; it < 2; ++it) {
                    (*weights[x])[z]->push_back(temp);
                }
            }
        }
    }
}

int main() {
    std::cout << "Working (currently)...";
    return 0;
}