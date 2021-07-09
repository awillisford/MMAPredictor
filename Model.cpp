#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

typedef unsigned int uint;

Model::Model(uint hiddenLayers, uint neuronsPerLayer, float lr) {
    // assign learning rate member to constructor parameter
    this->learningRate = lr;
    std::cout << "this->learningRate="<< this->learningRate << '\n';
    
    // create weight layer and gradient for input
    weights.push_back(new std::vector<std::vector<float>*>);
    nablaWeights.push_back(new std::vector<std::vector<float>*>);

    // create input weights 
    for (int x = 0; x < CsvToVector::features[0].size(); ++x) {
        weights[0]->push_back(new std::vector<float>);
        nablaWeights[0]->push_back(new std::vector<float>);
        for (int y = 0; y < neuronsPerLayer; ++y) {
            (*weights[0])[x]->push_back(0.5);
            (*nablaWeights[0])[x]->push_back(0.5);
        }
    }

    for (int x = 0; x < hiddenLayers; ++x) {
        weights.push_back(new std::vector<std::vector<float>*>);
        nablaWeights.push_back(new std::vector<std::vector<float>*>);
        cache.push_back(new std::vector<float>);
        nablaCache.push_back(new std::vector<float>);
        biases.push_back(new std::vector<float>);
        nablaBiases.push_back(new std::vector<float>);
        activated.push_back(new std::vector<float>);

        for (int y = 0; y < neuronsPerLayer; ++y) {
            weights[x + 1]->push_back(new std::vector<float>);
            nablaWeights[x + 1]->push_back(new std::vector<float>);
            cache[x]->push_back(0.5);
            nablaCache[x]->push_back(0.5);
            biases[x]->push_back(0.4);
            nablaBiases[x]->push_back(0.4);
            activated[x]->push_back(0.5);

            if (x < hiddenLayers - 1) {
                for (int z = 0; z < neuronsPerLayer; ++z) {
                    (*weights[x + 1])[y]->push_back(0.5);
                    (*nablaWeights[x + 1])[y]->push_back(0.5);
                }
            }
            else {
                for (int z = 0; z < 2; ++z) {
                    (*weights[x + 1])[y]->push_back(0.5);
                    (*nablaWeights[x + 1])[y]->push_back(0.5);
                }
            }
        }
        // last hidden layer
        if (x == hiddenLayers - 1) {
            // create members for output nodes
            cache.push_back(new std::vector<float>);
            nablaCache.push_back(new std::vector<float>);
            biases.push_back(new std::vector<float>);
            nablaBiases.push_back(new std::vector<float>);
            activated.push_back(new std::vector<float>);
            for (int y = 0; y < 2; ++y) {
                cache[x + 1]->push_back(0.5);
                nablaCache[x + 1]->push_back(0.5);
                biases[x + 1]->push_back(0.4);
                nablaBiases[x + 1]->push_back(0.4);
                activated[x + 1]->push_back(0.5);
            }
        }
    }
}

float Model::sigmoid(const float& in, bool derivative) {
    // using derivative when already passed through sigmoid function
    if (derivative == true) {
        return in * (1 - in);
    }

    return 1/(1 + std::exp(-in));
}

void Model::forward(const std::vector<float>& feature) {
    // zero values of each cache
    for (int x = 0; x < cache.size(); ++x) {
        for (int y = 0; y < cache[x]->size(); ++y) {
            (*cache[x])[y] = 0;
        }
    }

    // forward feed
    for (int x = 0; x < weights.size(); ++x) {
        for (int y = 0; y < weights[x]->size(); ++y) {
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // weights from input
                if (x == 0) {
                    (*cache[x])[z] += (*(*weights[x])[y])[z] * feature[y]; // add weight * value to cache value
                }
                // weights from hidden nodes
                else {
                    (*cache[x])[z] += (*(*weights[x])[y])[z] * (*activated[x - 1])[y]; // add weight * value to cache value
                }
                // last group of weights in layer
                if (y == weights[x]->size() - 1) {
                    (*cache[x])[z] += (*biases[x])[z]; // add biases
                    (*activated[x])[z] = sigmoid((*cache[x])[z]); // activated equal to cache through activation function
                }
            }
        }
    }
}

void Model::backward(int currentLabel) {
    // print label to user
    std::cout << "- [" << CsvToVector::labels[currentLabel][0] << ", "
                       << CsvToVector::labels[currentLabel][1] << "] label\n";
    // print loss to user
    std::cout << "- " << MSE(*activated.back(), CsvToVector::labels[currentLabel])
              << " loss\n";
    
    // start from end, weight layers
    for (int x = weights.size() - 1; x >= 0; x--) {
        
        if (x == weights.size() - 1) {
            // assign gradient of cache to partial derivative of activated value from output nodes
            (*nablaCache[x])[0] = sigmoid(MSE(*activated.back(), CsvToVector::labels[currentLabel], true, 0), true);
            (*nablaCache[x])[1] = sigmoid(MSE(*activated.back(), CsvToVector::labels[currentLabel], true, 1), true);
            // assign output node bias gradients to cache gradient of same nodes
            (*nablaBiases[x])[0] = (*nablaCache[x])[0];
            (*nablaBiases[x])[1] = (*nablaCache[x])[1];
        }
        // iterate through each weight vector in weight layer         
        for (int y = 0; y < weights[x]->size(); ++y) {
            float summationActivation = 0; // holds summation of partial deriv of loss with respect to activated node values
            
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // hidden weight layers
                if (x > 0) {
                    // set weight gradients from hidden layer output
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * (*activated[x - 1])[y];
                    // summation of weight * cache gradient
                    summationActivation += (*(*weights[x])[y])[z] * (*nablaCache[x])[z];
                }
                // input weight layer
                else {
                    // set weight gradients from input
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * CsvToVector::features[currentLabel][y];
                }
            }
            if (x > 0) {
                // set gradients of cache from activation for hidden layer nodes
                (*nablaCache[x - 1])[y] = sigmoid(summationActivation, true);
                // set gradients of biases from hidden layer nodes
                (*nablaBiases[x - 1])[y] = (*nablaCache[x - 1])[y];
            }
        }
    }

    // update weights by gradients
    for (int x = 0; x < weights.size(); ++x) {
        for (int y = 0; y < weights[x]->size(); ++y) {
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                (*(*weights[x])[y])[z] -= (*(*nablaWeights[x])[y])[z] * learningRate;
            }
        }
    }
    // update biases by gradients
    for (int x = 0; x < biases.size(); ++x) {
        for (int y = 0; y < biases[x]->size(); ++y) {
            (*biases[x])[y] -= (*nablaBiases[x])[y] * learningRate;
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

std::ostream& operator<<(std::ostream& out, Model& mod) {
    out << mod.str2(mod.nablaBiases, "nablaBiases")
    << mod.str2(mod.nablaCache, "nablaCache")
    << mod.str3(mod.nablaWeights, "nablaWeights")
    << mod.str2(mod.biases, "biases")
    << mod.str2(mod.cache, "cache")
    << mod.str3(mod.weights, "weights")
    << mod.str2(mod.activated, "activated");
    return out;
}

template <typename T> std::string Model::toStr(const T& t) {
    std::ostringstream os;
    os << t;
    return os.str();
}

std::string Model::str2(const std::vector<std::vector<float>*>& vec, const std::string name) {
    std::string str = "[";
    for (int x = 0; x < vec.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        for (int y = 0; y < vec[x]->size(); ++y) {
            if (y < vec[x]->size() - 1) {
                str += toStr((*vec[x])[y]);
                str += ", ";
            }
            else
                str += toStr((*vec[x])[y]);
        }
        if (x < vec.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(" + name + ")]\n\n";
    return str;
}

std::string Model::str3(const std::vector<std::vector<std::vector<float>*>*>& vec, const std::string name) {
    std::string str = "[";
    for (int x = 0; x < vec.size(); ++x) {
        if (x > 0)
            str += " [";
        else
            str += "[";
        int nodeLayerSize = vec[x]->size();
        for (int y = 0; y < nodeLayerSize; ++y) {
            str += "[";
            int nodeSize = (*vec[x])[y]->size();
            for (int z = 0; z < nodeSize; ++z) {
                if (z < nodeSize - 1) {
                    str += toStr((*(*vec[x])[y])[z]);
                    str += ", ";
                }
                else
                    str += toStr((*(*vec[x])[y])[z]);
            }
            if (y < nodeLayerSize - 1)
                str += "], ";
            else
                str += "]";
        }
        if (x < vec.size() - 1)
            str += "],\n";
        else
            str += "]";
    }
    str += "(" + name + ")]\n\n";
    return str;
}