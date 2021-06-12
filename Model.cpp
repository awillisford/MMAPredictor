#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>

Model::Model(uint numHiddenLayers, uint neuronsPerLayer, float learningRate) {
    // assign class attribute to constructor parameter
    this->learningRate = learningRate;

    // create layer and weight vector for input
    weights.push_back(new std::vector<std::vector<float>*>);
    layers.push_back(new std::vector<uint>);

    // create input nodes for layer vector and input weights 
    for (int it = 0; it < CsvToVector::features[0].size(); ++it) {
        layers[0]->push_back(it);
        weights[0]->push_back(new std::vector<float>);
    }

    // create hidden layer nodes and weights
    for (uint x = 0; x < numHiddenLayers; ++x) {
        // create hidden layers and weights
        layers.push_back(new std::vector<uint>); 
        weights.push_back(new std::vector<std::vector<float>*>);
        // node creation and weights vector per node
        for (uint y = 0; y < neuronsPerLayer; ++y) {
            layers[x + 1]->push_back(y);
            weights[x + 1]->push_back(new std::vector<float>);

            // create each weight float per nodes in next layer
            float temp;
            // if not currently on last hidden layer
            if (x < numHiddenLayers - 1) {
                for (uint z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
            // if on last layer
            else {
                // create two weights per node
                for (uint z = 0; z <= 1; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
        }
    }

    // create output layer nodes 
    layers.push_back(new std::vector<uint>);
    for (uint it = 0; it <= 2; ++it) {
        layers.back()->push_back(it);
    }
}

std::string Model::weightsToString() {
    std::string str;
    str += "["; // show beginning of weight vector

    // layers
    for (int z = 0; z < weights.size(); ++z) {
        str += "["; // show beginning of layer
        // 
        for (int x = 0; x < weights[z]->size(); ++x) {
            str += "["; // beginning of node
            // node vector 
            for (int y = 0; y < (*weights[z])[x]->size(); ++y) {
                // if not last element
                if (y < (*weights[z])[x]->size() - 1)
                    str += (*(*weights[z])[x])[y] , ", "; // end of current element, onto next
                // if last element
                else
                    str += (*(*weights[z])[x])[y];
            }
            // if not last node in layer
            if (x < weights[z]->size() - 1)
                str += "], "; // end of current node, onto next
            // last node in layer
            else
                str += "]";
        }
        // if not last layer
        if (z < weights.size() - 1)
            str += "],\n "; // end of current layer, onto next
        // last layer
        else 
            str += "]";
    }
    str += "]"; // show end of weight vector

    return str;
}