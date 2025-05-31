#pragma once
#include <vector>
#include <string>

namespace simulator {
class Device {
public:
    Device(double Lx, double Ly, const std::vector<std::map<std::string, double>>& regions = {});
    double get_epsilon_at(double x, double y) const;
    std::vector<double> get_extents() const { return {Lx_, Ly_}; }
private:
    double Lx_, Ly_;
    std::vector<std::map<std::string, double>> regions_;
};
} // namespace simulator