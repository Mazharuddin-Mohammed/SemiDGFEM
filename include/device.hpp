#pragma once
#include <vector>
#include <string>
#include <map>
#include <stdexcept>

namespace simulator {
class Device {
public:
    Device(double Lx, double Ly, const std::vector<std::map<std::string, double>>& regions = {});
    ~Device() = default;

    // Copy constructor and assignment operator
    Device(const Device& other) = default;
    Device& operator=(const Device& other) = default;

    // Move constructor and assignment operator
    Device(Device&& other) noexcept = default;
    Device& operator=(Device&& other) noexcept = default;

    double get_epsilon_at(double x, double y) const;
    std::vector<double> get_extents() const { return {Lx_, Ly_}; }

    // Validation methods
    bool is_valid() const { return Lx_ > 0.0 && Ly_ > 0.0; }
    void validate() const;

private:
    double Lx_, Ly_;
    std::vector<std::map<std::string, double>> regions_;
};

// C interface for Cython
extern "C" {
    Device* create_device(double Lx, double Ly);
    void destroy_device(Device* device);
    double device_get_epsilon_at(Device* device, double x, double y);
    void device_get_extents(Device* device, double* extents);
}

} // namespace simulator