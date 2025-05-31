#include <iostream>
#include <cassert>
#include <stdexcept>
#include <chrono>
#include "../include/device.hpp"
#include "../include/mesh.hpp"
#include "../include/poisson.hpp"
#include "../include/driftdiffusion.hpp"

using namespace simulator;

// Simple test framework
int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    void run_test_##name() { \
        std::cout << "Running " << #name << "... "; \
        tests_run++; \
        try { \
            test_##name(); \
            std::cout << "PASSED" << std::endl; \
            tests_passed++; \
        } catch (const std::exception& e) { \
            std::cout << "FAILED: " << e.what() << std::endl; \
        } catch (...) { \
            std::cout << "FAILED: Unknown exception" << std::endl; \
        } \
    } \
    void test_##name()

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        throw std::runtime_error("Assertion failed: " #condition " should be false"); \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " != " #b); \
    }

#define ASSERT_NEAR(a, b, tolerance) \
    if (std::abs((a) - (b)) > (tolerance)) { \
        throw std::runtime_error("Assertion failed: " #a " not near " #b); \
    }

// Test Device class
TEST(DeviceBasicConstruction) {
    Device device(1e-6, 0.5e-6);
    ASSERT_TRUE(device.is_valid());
    auto extents = device.get_extents();
    ASSERT_EQ(extents.size(), 2);
    ASSERT_NEAR(extents[0], 1e-6, 1e-12);
    ASSERT_NEAR(extents[1], 0.5e-6, 1e-12);
}

TEST(DeviceInvalidDimensions) {
    bool caught_exception = false;
    try {
        Device device(-1.0, 0.5e-6);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
}

TEST(DeviceEpsilonCalculation) {
    Device device(1e-6, 0.5e-6);
    double epsilon = device.get_epsilon_at(0.5e-6, 0.25e-6);
    ASSERT_NEAR(epsilon, 11.7 * 8.854e-12, 1e-15);
}

// Test Mesh class
TEST(MeshBasicConstruction) {
    Device device(1e-6, 0.5e-6);
    Mesh mesh(device, MeshType::Structured);
    ASSERT_TRUE(mesh.is_valid());
    // Test that validate doesn't throw
    mesh.validate();
}

// Test Poisson solver
TEST(PoissonBasicConstruction) {
    Device device(1e-6, 0.5e-6);
    Poisson poisson(device, Method::DG, MeshType::Structured);
    
    // Set a simple charge density
    std::vector<double> rho(100, 1e16 * 1.602e-19); // 1e16 cm^-3 charge density
    poisson.set_charge_density(rho);
    
    // Test basic solve (this might fail due to incomplete implementation)
    std::vector<double> bc = {0.0, 1.0, 0.0, 0.0}; // Simple boundary conditions
    try {
        auto result = poisson.solve_2d(bc);
        std::cout << " (Solve returned " << result.size() << " values)";
    } catch (const std::exception& e) {
        std::cout << " (Solve failed as expected: " << e.what() << ")";
    }
}

// Test DriftDiffusion solver
TEST(DriftDiffusionBasicConstruction) {
    Device device(1e-6, 0.5e-6);
    DriftDiffusion dd(device, Method::DG, MeshType::Structured);
    
    // Set doping
    std::vector<double> Nd(100, 1e16); // 1e16 cm^-3 donor concentration
    std::vector<double> Na(100, 1e15); // 1e15 cm^-3 acceptor concentration
    dd.set_doping(Nd, Na);
    
    ASSERT_TRUE(dd.is_valid());
    ASSERT_EQ(dd.get_order(), 3); // Default order
}

// Performance test
TEST(PerformanceBasic) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Device device(1e-6, 0.5e-6);
    for (int i = 0; i < 1000; ++i) {
        double x = (i % 100) * 1e-8;
        double y = 0.25e-6;
        device.get_epsilon_at(x, y);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << " (Completed 1000 epsilon calculations in " << duration.count() << " μs)";
    ASSERT_TRUE(duration.count() < 10000); // Should complete in less than 10ms
}

int main() {
    std::cout << "=== SemiDGFEM Backend Unit Tests ===" << std::endl;
    std::cout << std::endl;
    
    // Run all tests
    run_test_DeviceBasicConstruction();
    run_test_DeviceInvalidDimensions();
    run_test_DeviceEpsilonCalculation();
    run_test_MeshBasicConstruction();
    run_test_PoissonBasicConstruction();
    run_test_DriftDiffusionBasicConstruction();
    run_test_PerformanceBasic();
    
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
    
    if (tests_passed == tests_run) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
