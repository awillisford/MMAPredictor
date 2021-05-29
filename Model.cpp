#include "Model.h"
#include <iostream>

Model::Model(std::vector<float> input, uint numLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create vector for input layer's weights
    weights.push_back(new std::vector<std::vector<float>*>);
    // create a weight vector that holds type float for each node in input
    for (int it = 0; it < input.size() * 2; ++it) { // 2 inputs
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

    for (uint x = 0; x < numLayers + 1; ++x) {
        for (uint z = 0; z < neuronsPerLayer; ++z) {

            // FIGURE OUT WHY THIS WORKS
            for(int y = 0; y < neuronsPerLayer; ++y) {
                float temp = 1.0;
                weights[x][z][y]->push_back(temp);
            }

            // BUT THIS DOESNT?!?!?
            float temp = 1.0;
            weights[x][z].push_back(temp);

        }
    }
}

int main() {
    std::cout << "Working (currently)...";
    return 0;
}