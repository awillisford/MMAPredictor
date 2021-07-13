#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    std::vector<std::vector<std::vector<float>>> data = CsvToVector::extract_data("data/test.csv");
    std::vector<std::vector<float>> features = data[0];
    std::vector<std::vector<float>> labels = data[1];

    Model m1(features, 2, 3, .03);

    m1.randomize(); // randomize weights and biases

    int EPOCHS = 1;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int elem = 0; elem < features.size(); ++elem) {
            std::cout << m1 << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
            std::cout << features[0][0]<<", "<<features[0][1]<<", "<<features[0][2]<<'\n';
            m1.forward(features[elem]);
            std::cout << m1 << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
            m1.backward(features[elem], labels[elem]);
            std::cout << m1 << '\n';
            std::cout << "label: ["<<labels[0][0]<<", "<<labels[0][1]<<"]";
            return 1;
        }
        std::cout << "epoch(" << epoch << ") : ";
        m1.printLoss(features);
    }
} 