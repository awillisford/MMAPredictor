#pragma once

#include <vector>
#include <string>

class csvToVector {
    private:
    // TEMP COMMENT -- UPDATE
    static bool contains_number(const std::string& str);

    // TEMP COMMENT -- UPDATE 
    static float contains_text(const std::string& str);

    // TEMP COMMENT -- UPDATE
    static float check_cell_type(std::string& cell);

    public:
    // TEMP COMMENT -- UPDATE
    static std::vector<std::vector<float>> extract_data();
};
