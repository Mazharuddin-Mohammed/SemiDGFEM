#include <iostream>
#include <cassert>
#include <stdexcept>
#include "../include/device.hpp"

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

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " != " #b); \
    }

#define ASSERT_NEAR(a, b, tolerance) \
    if (std::abs((a) - (b)) > (tolerance)) { \
        throw std::runtime_error("Assertion failed: " #a " not near " #b); \
    }

// Test Device class basic functionality
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

TEST(DeviceMultipleInstances) {
    Device device1(1e-6, 0.5e-6);
    Device device2(2e-6, 1e-6);
    
    ASSERT_TRUE(device1.is_valid());
    ASSERT_TRUE(device2.is_valid());
    
    auto extents1 = device1.get_extents();
    auto extents2 = device2.get_extents();
    
    ASSERT_NEAR(extents1[0], 1e-6, 1e-12);
    ASSERT_NEAR(extents2[0], 2e-6, 1e-12);
}

int main() {
    std::cout << "=== SemiDGFEM Minimal Backend Tests ===" << std::endl;
    std::cout << std::endl;
    
    // Run all tests
    run_test_DeviceBasicConstruction();
    run_test_DeviceInvalidDimensions();
    run_test_DeviceEpsilonCalculation();
    run_test_DeviceMultipleInstances();
    
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
