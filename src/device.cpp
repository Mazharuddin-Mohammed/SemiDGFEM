#include "device.hpp"
#include <cmath>

namespace simulator {
Device::Device(double Lx, double Ly, const std::vector<std::map<std::string, double>>& regions)
    : Lx_(Lx), Ly_(Ly), regions_(regions) {}

double Device::get_epsilon_at(double x, double y) const {
    for (const auto& region : regions_) {
        if (x >= region.at("x_start") && x <= region.at("x_end") &&
            y >= region.at("y_start") && y <= region.at("y_end")) {
            return region.at("epsilon");
        }
    }
    return 11.7 * 8.854e-12;
}
} // namespace simulator