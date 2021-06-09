#include <fstream>
#include <sstream>
#include <iostream>
#include "../include/csvToVector.h"

bool CsvToVector::contains_number(const std::string& str) {
    /* string::npos is returned by method find when the digit/digits
     * were not found; therefore, if any of the digits are found, 
     * it will return that digit and be != to string::npos, which returns true */

    // return true if number found in string, else return false
    return (str.find_first_of("0123456789") != std::string::npos);
}

float CsvToVector::contains_text(const std::string& str) {
    // fighter color winner
    if (str == "Blue") {
        return 0;
    }
    else if (str == "Red") {
        return 1;
    }
    else if (str == "Draw") {
        return 2;
    }
    // fighter stance
    else if (str == "Orthodox") {
        return 0;
    }
    else if (str == "Southpaw") {
        return 1;
    }
    else if (str == "Switch") {
        return 2;
    }
    else if (str == "Open Stance") {
        return 3;
    }
    // title fight
    else if (str == "FALSE") {
        return 0;
    }
    else if (str == "TRUE") {
        return 1;
    }
    // weight divisions
    else if (str == "Flyweight") {
        return 0;
    }
    else if (str == "Bantamweight") {
        return 1;
    }
    else if (str == "Featherweight") {
        return 2;
    }
    else if (str == "Lightweight") {
        return 3;
    }
    else if (str == "Welterweight") {
        return 4;
    }
    else if (str == "Middleweight") {
        return 5;
    }
    else if (str == "LightHeavyweight") {
        return 6;
    }
    else if (str == "Heavyweight") {
        return 7;
    }
    else if (str == "CatchWeight") {
        return 8;
    }
    else if (str == "WomenStrawweight") {
        return 9;
    }
    else if (str == "WomenFlyweight") {
        return 10;
    }
    else if (str == "WomenBantamweight") {
        return 11;
    }
    else if (str == "WomenFeatherweight") {
        return 12;
    }

    // throw exception if text is not any of the above
    else {
        try {
            throw 1;
        }
        catch(int e) {
            std::cout << "An exception occured in function \"contains_text()\", exception no. "
                      << e << std::endl;
        }
        return -1;
    }
}

float CsvToVector::check_cell_type(std::string& cell) {
    // if cell contains a number return type float of string
    if (contains_number(cell)) {
        std::cout << "cell value: no. " << std::stof(cell) << std::endl;
        return std::stof(cell);
    }
    // return float value assigned to cell given by contains_text()
    else {
        std::cout << "cell value: text '" << cell << "'" << std::endl;
        return contains_text(cell);
    }
}

std::vector<std::vector<float>> CsvToVector::extract_data() {
    std::ifstream fin("data_filtered.csv"); // input stream
    std::string line; // declare string to hold each line

    std::vector<std::vector<float>> tokens;

    // loop through csv line by line
    while(std::getline(fin, line)) {
        std::vector<float> temp_vector; // create vector to push back in tokens
        std::stringstream ss(line);
        std::string cell; // holds data in cell

        // iterate through cells in line
        while(std::getline(ss, cell, ',')) { 
            temp_vector.push_back(check_cell_type(cell));
        }
        tokens.push_back(temp_vector); // push vector containing row into main vector
    }
    return tokens;
}

int main() {
    CsvToVector::extract_data();
}