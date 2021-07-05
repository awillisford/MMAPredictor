#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");

    Model m1(2, 2, 0.01);

    int EPOCHS = 3;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {
            // std::cout << "features size: " << CsvToVector::features.size() << '\n';
            // std::cout << "elem=" << elem << "\n";

            std::cout << m1.layersToString() << "\n\n";
            std::cout << m1.cacheToString() << "\n\n";
            std::cout << m1.activatedToString() << "\n\n";
            std::cout << m1.biasesToString() << "\n\n";
            std::cout << m1.nablaBToString() << "\n\n";
            std::cout << m1.nablaCacheToString() << "\n\n";
            std::cout << m1.nablaWToString() << "\n\n";
            std::cout << m1.weightsToString();
            
            m1.forward(CsvToVector::features[elem]);

            std::cout << "---------------FORWARD PASS-------------\n\n";
            std::cout << m1.layersToString() << "\n\n";
            std::cout << m1.cacheToString() << "\n\n";
            std::cout << m1.activatedToString() << "\n\n";
            std::cout << m1.biasesToString() << "\n\n";
            std::cout << m1.nablaBToString() << "\n\n";
            std::cout << m1.nablaCacheToString() << "\n\n";
            std::cout << m1.nablaWToString() << "\n\n";
            std::cout << m1.weightsToString();

            m1.backward(elem);

            std::cout << "---------------BACKWARD PASS-------------\n\n";
            std::cout << m1.layersToString() << "\n\n";
            std::cout << m1.cacheToString() << "\n\n";
            std::cout << m1.activatedToString() << "\n\n";
            std::cout << m1.biasesToString() << "\n\n";
            std::cout << m1.nablaBToString() << "\n\n";
            std::cout << m1.nablaCacheToString() << "\n\n";
            std::cout << m1.nablaWToString() << "\n\n";
            std::cout << m1.weightsToString();

            int t;
            std::cin >> t;
        }
    }
} 