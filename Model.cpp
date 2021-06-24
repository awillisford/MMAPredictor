#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <tuple>

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

    // initialize input weights
    float temp = 1.2;
    int numInputs = 0;
    for (int y = 0; y < weights[0]->size(); ++y) {
        for (int z = 0; z < neuronsPerLayer; ++z) {
            (*weights[0])[y]->push_back(temp);
        }
        numInputs++;
    }
    std::cout << "Inputs: " << numInputs << '\n';

    // create hidden layer nodes and weights
    for (uint x = 0; x < numHiddenLayers; ++x) {
        // create hidden layers and weights
        layers.push_back(new std::vector<uint>); 
        cache.push_back(new std::vector<float>);
        activated.push_back(new std::vector<float>);
        weights.push_back(new std::vector<std::vector<float>*>);
        // node creation and weights vector per node
        for (uint y = 0; y < neuronsPerLayer; ++y) {
            layers[x + 1]->push_back(y);
            cache[x]->push_back(y);
            activated[x]->push_back(y);
            weights[x + 1]->push_back(new std::vector<float>);

            // create each weight float per nodes in next layer
            float temp = 1.1;
            // if not currently on last hidden layer
            if (x < numHiddenLayers - 1) {
                for (uint z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
            // if on last layer
            else {
                // create two weights per node, since theres two outputs
                for (uint z = 0; z <= 1; ++z) {
                    (*weights[x + 1])[y]->push_back(temp);
                }
            }
        }
    }

    // create output layer nodes 
    layers.push_back(new std::vector<uint>);
    cache.push_back(new std::vector<float>);
    activated.push_back(new std::vector<float>);
    for (uint it = 0; it <= 1; ++it) { // two iterations for two outputs
        layers.back()->push_back(it);
        cache.back()->push_back(it);
        activated.back()->push_back(it);
    }
    // initialize biases
    init_biases();

    // initialize deltas
    deltas = biases;
}

float Model::sigmoid(const float& in, bool derivative) {
    // using derivative when already passed through sigmoid function
    if (derivative == true) {
        return in * (1 - in);
    }

    return 1/(1 + std::exp(-in));
}

void Model::forward(const std::vector<float>& feature) {
    for (int x = 0; x < layers.size(); ++x) {
        // skip first iteration, I started at zero because it makes readability easier
        if (x == 0) {
            continue;
        }

        for (int y = 0; y < layers[x]->size(); ++y) {
            float sum = 0;
            // iterate through weights that connect to current node
            for (int it = 0; it < weights[x - 1]->size(); ++it) {
                if (x == 1) {
                    sum += (*(*weights[x - 1])[it])[y] * feature[it];
                }
                else {
                    sum += (*(*weights[x - 1])[it])[y] * (*activated[x - 1])[y];
                }
            }
            // assign cache values to sum of previous outputs * weights + bias
            (*cache[x - 1])[y] = sum + (*biases[x - 1])[y];
            // assign activated values to cache put through sigmoid function
            (*activated[x - 1])[y] = sigmoid((*cache[x - 1])[y]);
        }
    }
}

float Model::MSE(std::vector<float> output, std::vector<float> label, bool derivative, int element) {
    if (derivative == true) {
        return output[element] - label[element];
    }
    float sum = 0;
    int size = output.size();
    for (int x = 0; x < size; ++x) {
        float squared = std::pow(label[x] - output[x], 2) ;
        sum += squared;
    }
    return (sum / size);
}

void Model::init_biases() {
    for (int x = 0; x < layers.size(); ++x) {
        // if not input layer
        if (x > 0) {
            biases.push_back(new std::vector<float>);
            // add bias element for each node
            for (int y = 0; y < layers[x]->size(); ++y) {
                float temp = 1.5;
                biases[x - 1]->push_back(temp);
            }
        }
        // skip applying biases for input layer
        else {
            continue;
        }
    }
}

float Model::getActivated(std::tuple<int, int> position) {
    return (*activated[std::get<0>(position)])[std::get<1>(position)];
}

template <typename T> std::string Model::toStr(const T& t) { 
   std::ostringstream os; 
   os << t; 
   return os.str(); 
} 

std::string Model::weightsToString() {
    std::string str;
    str += "["; // show beginning of weight vector

    // weight layers
    for (int x = 0; x < weights.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        // node weight vector
        int nodeLayerSize = weights[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            // individual weights
            int nodeSize = (*weights[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                // if not last element
                if (z < nodeSize - 1) {
                    str += toStr((*(*weights[x])[y])[z]);
                    str += ", ";
                }
                // if last element
                else {
                    str += toStr((*(*weights[x])[y])[z]);
                }
            }
            // if not last element
            if (y < nodeLayerSize - 1) {
                str += "], ";
            }
            // if last element
            else {
                str += "]";
            }
        }
        // if not last element
        if (x < weights.size() - 1) {
            str += "],\n";
        }
        // if last element
        else {
            str += "]";
        }
    }
    str += "(weights)]";

    return str;
}

std::string Model::layersToString() {
    std::string str;
    str += "[";

    for (int x = 0; x < layers.size(); ++x) {
        if (x == 0)
            str += "[";
        else
            str += " [";
        
        for (int y = 0; y < layers[x]->size(); ++y) {
            if (y < layers[x]->size() - 1) {
                str += toStr((*layers[x])[y]);
                str += ", ";
            }
            else
                str += toStr((*layers[x])[y]);
        }
        if (x < layers.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(layers)]";

    return str;
}

std::string Model::biasesToString() {
    std::string str;
    str += "[";

    for (int x = 0; x < biases.size(); ++x) {
        if (x == 0)
            str += "[";
        else
            str += " [";
        
        for (int y = 0; y < biases[x]->size(); ++y) {
            if (y < biases[x]->size() - 1) {
                str += toStr((*biases[x])[y]);
                str += ", ";
            }
            else
                str += toStr((*biases[x])[y]);
        }
        if (x < biases.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(biases)]";

    return str;
}