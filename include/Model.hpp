#pragma once

#include <vector>
#include <string>

typedef unsigned int uint;

class Model {
    private:
    // attributes 
    std::vector<std::vector<uint>*> layers; // holds nodes
    std::vector<std::vector<float>*> cache; // holds unactivated values of node
    std::vector<std::vector<float>*> activated; // holds activated values of nodes
    std::vector<std::vector<float>*> biases; // holds biases for nodes, except input
    std::vector<std::vector<float>*> gradientsB; // holds gradient for each bias
    std::vector<std::vector<std::vector<float>*>*> gradientsW; // holds gradient for each weight
    std::vector<std::vector<std::vector<float>*>*> weights;
    uint learningRate;

    // initializes biases
    void init_biases();

    // used for changing float to string
    template <typename T> std::string toStr(const T& t);

    // sigmoid activation function
    float sigmoid(const float& in, bool derivative=false);

    // return activated output from given node position
    float getActivated(const int x, const int y);

    // mean squared error function for calculating loss
    float MSE(std::vector<float> output, std::vector<float> label, bool derivative=false, int element=0);
 
    public:
    // constructor
    Model(uint numLayers, uint neuronsPerLayer, float learningRate = 0.05);

    // feed input forward and get output
    void forward(const std::vector<float>& feature);

    // backpropagate to compute gradients for each weight and bias
    void backward(int currentLabel);

    void zeroGradients();
    
    // weight vector to string for printing to console
    std::string weightsToString();

    // layer vector to string for printing to console
    std::string layersToString();

    // biases vector to string for printing to console
    std::string biasesToString();

    // activated vector to string for printing to console
    std::string activatedToString();
};