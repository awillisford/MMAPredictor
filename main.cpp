#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");
    Model m1(4, 10);

    m1.randomize(); // randomize weights and biases

    int EPOCHS = 10;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {
            m1.forward(CsvToVector::features[elem]);
            m1.backward(elem);
        }
        std::cout << "epoch(" << epoch << ") : ";
        m1.printLoss();
        std::cout << m1;
    }
} 