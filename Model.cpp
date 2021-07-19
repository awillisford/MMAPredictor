#include "include/Model.hpp"
#include "include/CsvToVector.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <time.h> // randomize seed
#include <stdlib.h> // RANDOMIZE

typedef unsigned int uint;

Model::Model(std::vector<std::vector<float>>& features, uint hiddenLayers, uint neuronsPerLayer, float lr) {
    // assign learning rate member to constructor parameter
    this->learningRate = lr;
    std::cout << "learningRate=" << learningRate << '\n';

    std::cout << "individual feature size=" << features[0].size() << '\n';

    float summationLoss = 0;
    
    // create weight layer and gradient for input
    weights.push_back(new std::vector<std::vector<float>*>);
    nablaWeights.push_back(new std::vector<std::vector<float>*>);

    // create input weights 
    for (int x = 0; x < features[0].size(); ++x) {
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
            biases[x]->push_back(0.5);
            nablaBiases[x]->push_back(0.5);
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
                biases[x + 1]->push_back(0.5);
                nablaBiases[x + 1]->push_back(0.5);
                activated[x + 1]->push_back(0.5);
            }
        }
    }
}

float Model::sigmoid(const float& in, bool derivative) {
    // using derivative when already passed through sigmoid function
    if (derivative == true)  {
        // std::cout << "in * (1 - in) = " << in <<" * ("<< 1 <<" - "<< in << ")\n";
        return in * (1 - in);
    }
    return 1/(1 + std::exp(-in));
}

float Model::ReLU(const float& in) {
    /* Doesnt need a derivative parameter since any value greater than zero 
     * has a derivative of one, values zero or less have a derivative of zero */
    return (in > 0) ? in : 0;
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
                    (*cache[x])[z] += (*biases[x])[z]; // add biases to cache
                    // not last weight layer
                    if (x != weights.size() - 1) {
                        (*activated[x])[z] = sigmoid((*cache[x])[z]); // activated equal to cache through activation function
                    }
                    // last weight layer
                    // else {
                    //     (*activated[x])[z] = sigmoid((*cache[x])[z]); // activated equal to cache through activation function
                    // }
                }
            }
        }
    }
    *activated.back() = softmax(*cache.back());
}

void Model::backward(const std::vector<float>& feature, const std::vector<float>& label) {
    float loss = crossEntropy(*activated.back(), label);
    // std::cout <<"activated=["<<(*activated.back())[0]<<", "<<(*activated.back())[1]<<"] - label=["<<label[0]<<", "<<label[1]<<"]\n";
    // std::cout << "loss=" << loss << '\n';

    if (argmax(*activated.back()) == label)
        correct++;

    summationLoss += loss;

    // start from end, weight layers
    for (int x = weights.size() - 1; x >= 0; x--) {
        if (x == weights.size() - 1) {
            // assign gradient of cache to partial derivative of activated value from output nodes
            (*nablaCache[x])[0] = crossEntropy(*activated.back(), label, true, 0);
            // std::cout << "(*nablaCache[x])[0]="<<(*nablaCache[x])[0]<<'\n';
            (*nablaCache[x])[1] = crossEntropy(*activated.back(), label, true, 1);
            // std::cout << "(*nablaCache[x])[1]="<<(*nablaCache[x])[1]<<'\n';
            (*nablaBiases[x])[0] = (*nablaCache[x])[0];
            (*nablaBiases[x])[1] = (*nablaCache[x])[1];
        }
        // iterate through each weight vector in weight layer         
        for (int y = 0; y < weights[x]->size(); ++y) {
            float summationActivation = 0; // holds summation of partial deriv of loss with respect to activated node values
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                // hidden weight layers
                if (x > 0) {
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * (*activated[x - 1])[y];
                    summationActivation += (*(*weights[x])[y])[z] * (*nablaCache[x])[z];
                }
                // input weight layer
                else {
                    (*(*nablaWeights[x])[y])[z] = (*nablaCache[x])[z] * feature[y];
                }
            }
            if (x > 0) {
                (*nablaCache[x - 1])[y] = sigmoid(summationActivation, true);
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
        // std::cout << "output[element] - label[element] = "<< output[element] << " - " << label[element] << '\n';
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

std::vector<float> Model::argmax(std::vector<float> output) {
    if (output[0] > output[1]) {
        output[0] = 1;
        output[1] = 0;
    }
    else {
        output[0] = 0;
        output[1] = 1;
    }
    return output;
}

float Model::crossEntropy(std::vector<float> output, std::vector<float> label, bool derivative, int element) {
    // only use with softmax! //

    if (derivative == true) {
        if (label[element] == 1) {
            return output[element] - 1;
        }
        else {
            return output[element];
        }
    }
    return label[0] == 1 ? -log(output[0]) : -log(output[1]);
}

std::vector<float> Model::softmax(std::vector<float> output) {
    output[0] = exp(output[0])/(exp(output[0]) + exp(output[1]));
    output[1] = exp(output[1])/(exp(output[0]) + exp(output[1]));
    return output;
}

std::ostream& operator<<(std::ostream& out, Model& mod) {
    out << mod.str2(mod.nablaBiases, "nablaBiases")
        << mod.str2(mod.nablaCache, "nablaCache")
        << mod.str3(mod.nablaWeights, "nablaWeights")
        << mod.str3(mod.weights, "weights")
        << mod.str2(mod.biases, "biases")
        << mod.str2(mod.cache, "cache")
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

void Model::printLoss(std::vector<std::vector<float>>& features) {
    std::cout << "avg loss=" << (summationLoss / features.size()) << " : " << (float) correct * 100 / features.size() << "%\n";
    correct = 0;
    summationLoss = 0;
}

void Model::randomize() {
    srand(time(NULL));
    // randomize weights
    for (int x = 0; x < weights.size(); ++x) {
        for (int y = 0; y < weights[x]->size(); ++y) {
            for (int z = 0; z < (*weights[x])[y]->size(); ++z) {
                (*(*weights[x])[y])[z] = (float) rand() / (float) RAND_MAX;
            }
        }
    }
    // randomize biases
    for (int x = 0; x < biases.size(); ++x) {
        for (int y = 0; y < biases[x]->size(); ++y) {
            (*biases[x])[y] = (float) rand() / (float) RAND_MAX;
        }
    }
}