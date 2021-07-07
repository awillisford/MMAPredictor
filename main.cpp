#include <iostream>
#include <vector>
#include "include/CsvToVector.hpp"
#include "include/Model.hpp"

int main() {
    CsvToVector::extract_data("data/data_filtered.csv");

    Model m1(1, 2, 0.05);

    int EPOCHS = 3;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (int elem = 0; elem < CsvToVector::features.size(); ++elem) {

            std::cout << "---------------BEFORE FORWARD PASS-------------\n";
            std::cout << m1.nablaWToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.activatedToString() << "\n";

            double sum = 0;
            for (int x = 0; x < CsvToVector::features[elem].size(); ++x) {
                sum += CsvToVector::features[elem][x];
            }
            std::cout << "input summation: " << sum << "\n";

            m1.forward(CsvToVector::features[elem]);

            std::cout << "---------------AFTER FORWARD PASS-------------\n";
            std::cout << m1.nablaWToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.activatedToString() << "\n";

            m1.backward(elem);

            std::cout << "---------------AFTER BACKWARD PASS-------------\n";
            std::cout << m1.nablaWToString() << "\n";
            std::cout << m1.weightsToString() << "\n";
            std::cout << m1.nablaCacheToString() << "\n";
            std::cout << m1.cacheToString() << "\n";
            std::cout << m1.nablaBToString() << "\n";
            std::cout << m1.biasesToString() << "\n";
            std::cout << m1.activatedToString() << "\n";

            int t;
            std::cin >> t;
        }
    }
} 