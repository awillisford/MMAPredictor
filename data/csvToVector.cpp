#include <fstream>
#include <sstream>
#include <iostream>
#include "../include/csvToVector.h"

bool csvToVector::contains_number(const std::string& str) {
    /* string::npos is returned by method find when the digit/digits
     * were not found; therefore, if any of the digits are found, 
     * it will return that digit and be != to string::npos, which returns true */

    // return true if number found in string, else return false
    return (str.find_first_of("0123456789") != std::string::npos);
}

float csvToVector::contains_text(const std::string& str) {
    // assign float values to each of the different text
    if (str == "Orthodox") {
        return 1.0;
    }
    else if (str == "Southpaw") {
        return 2.0;
    }
    else if (str == "Switch") {
        return 3.0;
    }
    else if (str == "FALSE") {
        return 0.0;
    }
    else if (str == "TRUE") {
        return 1.0;
    }

    // throw exception if text is not any of the above
    else {
        try {
            throw 1;
        }
        catch(int e) {
            std::cout << "An exception occured in function \"contains_text\", exception no. "
                      << e << std::endl;
        }
    }
}

float csvToVector::check_cell_type(std::string& cell) {
    // if cell contains a number return type float of string
    if (contains_number(cell)) {
        std::cout << "cell contains number" << " ";
        return std::stof(cell);
    }
    // return float value assigned to cell given by contains_text()
    else {
        return contains_text(cell);
    }
}

std::vector<std::vector<float>> csvToVector::extract_data() {
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
    csvToVector::extract_data();
}