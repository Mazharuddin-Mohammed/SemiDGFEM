/**
 * Device geometry and material properties implementation
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "device.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace simulator {

Device::Device(double Lx, double Ly, const std::vector<std::map<std::string, double>>& regions)
    : Lx_(Lx), Ly_(Ly), regions_(regions) {
    if (Lx <= 0.0 || Ly <= 0.0) {
        throw std::invalid_argument("Device dimensions must be positive");
    }
    validate();
}

void Device::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("Device is in invalid state");
    }

    // Validate regions
    for (const auto& region : regions_) {
        const std::vector<std::string> required_keys = {"x_start", "x_end", "y_start", "y_end", "epsilon"};
        for (const auto& key : required_keys) {
            if (region.find(key) == region.end()) {
                throw std::invalid_argument("Region missing required key: " + key);
            }
        }

        double x_start = region.at("x_start");
        double x_end = region.at("x_end");
        double y_start = region.at("y_start");
        double y_end = region.at("y_end");
        double epsilon = region.at("epsilon");

        if (x_start >= x_end || y_start >= y_end) {
            throw std::invalid_argument("Invalid region bounds");
        }
        if (epsilon <= 0.0) {
            throw std::invalid_argument("Epsilon must be positive");
        }
        if (x_start < 0.0 || x_end > Lx_ || y_start < 0.0 || y_end > Ly_) {
            throw std::invalid_argument("Region extends beyond device bounds");
        }
    }
}

double Device::get_epsilon_at(double x, double y) const {
    // Bounds checking
    if (x < 0.0 || x > Lx_ || y < 0.0 || y > Ly_) {
        throw std::out_of_range("Coordinates outside device bounds");
    }

    for (const auto& region : regions_) {
        try {
            if (x >= region.at("x_start") && x <= region.at("x_end") &&
                y >= region.at("y_start") && y <= region.at("y_end")) {
                return region.at("epsilon");
            }
        } catch (const std::out_of_range& e) {
            // Skip malformed regions
            continue;
        }
    }

    // Default silicon permittivity
    return 11.7 * 8.854e-12;
}

// C interface implementations
extern "C" {
    Device* create_device(double Lx, double Ly) {
        try {
            return new Device(Lx, Ly);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_device(Device* device) {
        delete device;
    }

    double device_get_epsilon_at(Device* device, double x, double y) {
        if (!device) return 0.0;
        try {
            return device->get_epsilon_at(x, y);
        } catch (...) {
            return 11.7 * 8.854e-12; // Default value
        }
    }

    void device_get_extents(Device* device, double* extents) {
        if (!device || !extents) return;
        try {
            auto ext = device->get_extents();
            extents[0] = ext[0];
            extents[1] = ext[1];
        } catch (...) {
            extents[0] = extents[1] = 0.0;
        }
    }
}

} // namespace simulator