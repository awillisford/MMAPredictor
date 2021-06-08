#pragma once

#include <vector>
#include <string>

class Data {
    private:
    static bool contains_number(const std::string& str);

    static float contains_text(const std::string& str);

    static float check_cell_type(std::string& cell);

    public:
    static std::vector<std::vector<float>> extract_data();
};
