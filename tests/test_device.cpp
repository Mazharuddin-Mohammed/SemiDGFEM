#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "device.hpp"
#include <stdexcept>

using namespace simulator;

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
    }
    
    void TearDown() override {
        // Clean up
    }
};

// Test basic device construction
TEST_F(DeviceTest, BasicConstruction) {
    EXPECT_NO_THROW({
        Device device(1e-6, 0.5e-6);
        EXPECT_TRUE(device.is_valid());
        auto extents = device.get_extents();
        EXPECT_DOUBLE_EQ(extents[0], 1e-6);
        EXPECT_DOUBLE_EQ(extents[1], 0.5e-6);
    });
}

// Test invalid dimensions
TEST_F(DeviceTest, InvalidDimensions) {
    EXPECT_THROW(Device(-1.0, 0.5e-6), std::invalid_argument);
    EXPECT_THROW(Device(1e-6, -0.5e-6), std::invalid_argument);
    EXPECT_THROW(Device(0.0, 0.5e-6), std::invalid_argument);
    EXPECT_THROW(Device(1e-6, 0.0), std::invalid_argument);
}

// Test device with regions
TEST_F(DeviceTest, DeviceWithRegions) {
    std::vector<std::map<std::string, double>> regions = {
        {{"x_start", 0.0}, {"x_end", 0.5e-6}, {"y_start", 0.0}, {"y_end", 0.25e-6}, {"epsilon", 12.0 * 8.854e-12}},
        {{"x_start", 0.5e-6}, {"x_end", 1e-6}, {"y_start", 0.25e-6}, {"y_end", 0.5e-6}, {"epsilon", 3.9 * 8.854e-12}}
    };
    
    EXPECT_NO_THROW({
        Device device(1e-6, 0.5e-6, regions);
        EXPECT_TRUE(device.is_valid());
    });
}

// Test invalid regions
TEST_F(DeviceTest, InvalidRegions) {
    // Missing required key
    std::vector<std::map<std::string, double>> invalid_regions1 = {
        {{"x_start", 0.0}, {"x_end", 0.5e-6}, {"y_start", 0.0}, {"y_end", 0.25e-6}} // Missing epsilon
    };
    EXPECT_THROW(Device(1e-6, 0.5e-6, invalid_regions1), std::invalid_argument);
    
    // Invalid bounds
    std::vector<std::map<std::string, double>> invalid_regions2 = {
        {{"x_start", 0.5e-6}, {"x_end", 0.0}, {"y_start", 0.0}, {"y_end", 0.25e-6}, {"epsilon", 12.0 * 8.854e-12}} // x_start > x_end
    };
    EXPECT_THROW(Device(1e-6, 0.5e-6, invalid_regions2), std::invalid_argument);
    
    // Region extends beyond device
    std::vector<std::map<std::string, double>> invalid_regions3 = {
        {{"x_start", 0.0}, {"x_end", 2e-6}, {"y_start", 0.0}, {"y_end", 0.25e-6}, {"epsilon", 12.0 * 8.854e-12}} // x_end > device width
    };
    EXPECT_THROW(Device(1e-6, 0.5e-6, invalid_regions3), std::invalid_argument);
    
    // Negative epsilon
    std::vector<std::map<std::string, double>> invalid_regions4 = {
        {{"x_start", 0.0}, {"x_end", 0.5e-6}, {"y_start", 0.0}, {"y_end", 0.25e-6}, {"epsilon", -1.0}}
    };
    EXPECT_THROW(Device(1e-6, 0.5e-6, invalid_regions4), std::invalid_argument);
}

// Test epsilon calculation
TEST_F(DeviceTest, EpsilonCalculation) {
    std::vector<std::map<std::string, double>> regions = {
        {{"x_start", 0.0}, {"x_end", 0.5e-6}, {"y_start", 0.0}, {"y_end", 0.5e-6}, {"epsilon", 12.0 * 8.854e-12}}
    };
    
    Device device(1e-6, 0.5e-6, regions);
    
    // Point inside region
    EXPECT_DOUBLE_EQ(device.get_epsilon_at(0.25e-6, 0.25e-6), 12.0 * 8.854e-12);
    
    // Point outside region (should return default silicon permittivity)
    EXPECT_DOUBLE_EQ(device.get_epsilon_at(0.75e-6, 0.25e-6), 11.7 * 8.854e-12);
    
    // Point at boundary
    EXPECT_DOUBLE_EQ(device.get_epsilon_at(0.5e-6, 0.5e-6), 12.0 * 8.854e-12);
}

// Test bounds checking
TEST_F(DeviceTest, BoundsChecking) {
    Device device(1e-6, 0.5e-6);
    
    // Valid coordinates
    EXPECT_NO_THROW(device.get_epsilon_at(0.5e-6, 0.25e-6));
    
    // Invalid coordinates
    EXPECT_THROW(device.get_epsilon_at(-0.1e-6, 0.25e-6), std::out_of_range);
    EXPECT_THROW(device.get_epsilon_at(1.1e-6, 0.25e-6), std::out_of_range);
    EXPECT_THROW(device.get_epsilon_at(0.5e-6, -0.1e-6), std::out_of_range);
    EXPECT_THROW(device.get_epsilon_at(0.5e-6, 0.6e-6), std::out_of_range);
}

// Test C interface
TEST_F(DeviceTest, CInterface) {
    // Test creation
    Device* device = create_device(1e-6, 0.5e-6);
    ASSERT_NE(device, nullptr);
    
    // Test epsilon calculation
    double epsilon = device_get_epsilon_at(device, 0.5e-6, 0.25e-6);
    EXPECT_DOUBLE_EQ(epsilon, 11.7 * 8.854e-12);
    
    // Test extents
    double extents[2];
    device_get_extents(device, extents);
    EXPECT_DOUBLE_EQ(extents[0], 1e-6);
    EXPECT_DOUBLE_EQ(extents[1], 0.5e-6);
    
    // Test null pointer handling
    EXPECT_EQ(device_get_epsilon_at(nullptr, 0.5e-6, 0.25e-6), 0.0);
    device_get_extents(nullptr, extents);
    device_get_extents(device, nullptr);
    
    // Clean up
    destroy_device(device);
    destroy_device(nullptr); // Should not crash
}

// Test invalid device creation through C interface
TEST_F(DeviceTest, CInterfaceInvalidCreation) {
    Device* device = create_device(-1.0, 0.5e-6);
    EXPECT_EQ(device, nullptr);
    
    device = create_device(1e-6, -0.5e-6);
    EXPECT_EQ(device, nullptr);
}

// Test copy and move semantics
TEST_F(DeviceTest, CopyMoveSemantics) {
    Device original(1e-6, 0.5e-6);
    
    // Copy constructor
    Device copied(original);
    EXPECT_TRUE(copied.is_valid());
    auto extents = copied.get_extents();
    EXPECT_DOUBLE_EQ(extents[0], 1e-6);
    EXPECT_DOUBLE_EQ(extents[1], 0.5e-6);
    
    // Copy assignment
    Device assigned(2e-6, 1e-6);
    assigned = original;
    extents = assigned.get_extents();
    EXPECT_DOUBLE_EQ(extents[0], 1e-6);
    EXPECT_DOUBLE_EQ(extents[1], 0.5e-6);
    
    // Move constructor
    Device moved(std::move(original));
    EXPECT_TRUE(moved.is_valid());
    extents = moved.get_extents();
    EXPECT_DOUBLE_EQ(extents[0], 1e-6);
    EXPECT_DOUBLE_EQ(extents[1], 0.5e-6);
}

// Performance test
TEST_F(DeviceTest, PerformanceTest) {
    std::vector<std::map<std::string, double>> regions;
    for (int i = 0; i < 100; ++i) {
        double x_start = i * 1e-8;
        double x_end = (i + 1) * 1e-8;
        regions.push_back({
            {"x_start", x_start}, {"x_end", x_end}, 
            {"y_start", 0.0}, {"y_end", 0.5e-6}, 
            {"epsilon", (11.0 + i % 5) * 8.854e-12}
        });
    }
    
    Device device(1e-6, 0.5e-6, regions);
    
    // Time epsilon calculations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        double x = (i % 1000) * 1e-9;
        double y = 0.25e-6;
        device.get_epsilon_at(x, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete in reasonable time (less than 100ms)
    EXPECT_LT(duration.count(), 100000);
}
