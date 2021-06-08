#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

int main() {
    std::ifstream fin("data_filtered.csv");
    std::string line; // declare string to hold each line
    
    std::vector<std::string> tokens;

    while (std::getline(fin, line)) {
        // std::cout << line << std::endl;
        std::stringstream check1(line);
        std::string intermediate;

        while(std::getline(check1, intermediate, ',')) {
            tokens.push_back(intermediate);
            std::cout << intermediate << " ";
        }
        std::cout << std::endl;
    }

    fin.close();

    return 0;
}