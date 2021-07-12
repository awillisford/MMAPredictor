#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    std::vector<std::vector<std::vector<float>>> data = CsvToVector::extract_data("data/data_filtered.csv");
    std::vector<std::vector<float>> features = data[0];
    std::vector<std::vector<float>> labels = data[1];

    Model m1(features, 4, 40, .01);

    m1.randomize(); // randomize weights and biases

    std::cout << m1 << '\n';

    int EPOCHS = 1;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int elem = 0; elem < features.size(); ++elem) {
            m1.forward(features[elem]);
            // std::cout << m1;
            m1.backward(features[elem], labels[elem]);
        }
        std::cout << "epoch(" << epoch << ") : ";
        m1.printLoss(features);
        std::cout << m1;
    }
} 