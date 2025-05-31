#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "mesh.hpp"
#include "device.hpp"
#include <stdexcept>
#include <fstream>

using namespace simulator;

class MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = std::make_unique<Device>(1e-6, 0.5e-6);
    }
    
    void TearDown() override {
        // Clean up test files
        std::remove("test_mesh.msh");
        std::remove("device_refined.msh");
    }
    
    std::unique_ptr<Device> device_;
};

// Test structured mesh creation
TEST_F(MeshTest, StructuredMeshCreation) {
    EXPECT_NO_THROW({
        Mesh mesh(*device_, MeshType::Structured);
        EXPECT_TRUE(mesh.is_valid());
        EXPECT_GT(mesh.get_num_nodes(), 0);
        EXPECT_GT(mesh.get_num_elements(), 0);
        EXPECT_EQ(mesh.get_grid_points_x().size(), mesh.get_grid_points_y().size());
    });
}

// Test unstructured mesh creation
TEST_F(MeshTest, UnstructuredMeshCreation) {
    EXPECT_NO_THROW({
        Mesh mesh(*device_, MeshType::Unstructured);
        // Unstructured mesh starts empty until GMSH generation
        EXPECT_EQ(mesh.get_num_nodes(), 0);
        EXPECT_EQ(mesh.get_num_elements(), 0);
    });
}

// Test invalid device
TEST_F(MeshTest, InvalidDevice) {
    Device invalid_device(-1.0, 0.5e-6); // This should throw in constructor
    // But if we somehow get an invalid device, mesh should handle it
    // We'll test this through the C interface
}

// Test GMSH mesh generation
TEST_F(MeshTest, GMSHMeshGeneration) {
    Mesh mesh(*device_, MeshType::Unstructured);
    
    EXPECT_NO_THROW({
        mesh.generate_gmsh_mesh("test_mesh.msh");
        EXPECT_TRUE(mesh.is_valid());
        EXPECT_GT(mesh.get_num_nodes(), 0);
        EXPECT_GT(mesh.get_num_elements(), 0);
    });
    
    // Check if file was created
    std::ifstream file("test_mesh.msh");
    EXPECT_TRUE(file.good());
    file.close();
}

// Test GMSH with invalid parameters
TEST_F(MeshTest, GMSHInvalidParameters) {
    Mesh structured_mesh(*device_, MeshType::Structured);
    
    // Should throw for structured mesh
    EXPECT_THROW(structured_mesh.generate_gmsh_mesh("test.msh"), std::invalid_argument);
    
    Mesh unstructured_mesh(*device_, MeshType::Unstructured);
    
    // Should throw for empty filename
    EXPECT_THROW(unstructured_mesh.generate_gmsh_mesh(""), std::invalid_argument);
}

// Test mesh refinement
TEST_F(MeshTest, MeshRefinement) {
    Mesh mesh(*device_, MeshType::Structured);
    
    size_t initial_elements = mesh.get_num_elements();
    size_t initial_nodes = mesh.get_num_nodes();
    
    // Create refinement flags (refine every other element)
    std::vector<bool> refine_flags(initial_elements, false);
    for (size_t i = 0; i < initial_elements; i += 2) {
        refine_flags[i] = true;
    }
    
    EXPECT_NO_THROW({
        mesh.refine(refine_flags);
        EXPECT_TRUE(mesh.is_valid());
        EXPECT_GT(mesh.get_num_elements(), initial_elements);
        EXPECT_GT(mesh.get_num_nodes(), initial_nodes);
    });
}

// Test refinement with invalid flags
TEST_F(MeshTest, RefinementInvalidFlags) {
    Mesh mesh(*device_, MeshType::Structured);
    
    // Wrong size refinement flags
    std::vector<bool> wrong_size_flags(mesh.get_num_elements() + 10, false);
    EXPECT_THROW(mesh.refine(wrong_size_flags), std::invalid_argument);
    
    std::vector<bool> too_small_flags(mesh.get_num_elements() - 10, false);
    EXPECT_THROW(mesh.refine(too_small_flags), std::invalid_argument);
}

// Test mesh quality metrics
TEST_F(MeshTest, MeshQualityMetrics) {
    Mesh mesh(*device_, MeshType::Structured);
    
    double min_quality = mesh.get_min_element_quality();
    double max_size = mesh.get_max_element_size();
    
    EXPECT_GE(min_quality, 0.0);
    EXPECT_LE(min_quality, 1.0);
    EXPECT_GT(max_size, 0.0);
    
    // For structured mesh, quality should be reasonable
    EXPECT_GT(min_quality, 0.1); // Structured triangular mesh should have decent quality
}

// Test element connectivity validation
TEST_F(MeshTest, ElementConnectivityValidation) {
    Mesh mesh(*device_, MeshType::Structured);
    
    // Mesh should be valid after construction
    EXPECT_NO_THROW(mesh.validate());
    
    auto elements = mesh.get_elements();
    for (const auto& element : elements) {
        EXPECT_EQ(element.size(), 3); // Triangular elements
        for (int node_id : element) {
            EXPECT_GE(node_id, 0);
            EXPECT_LT(node_id, static_cast<int>(mesh.get_num_nodes()));
        }
    }
}

// Test grid point access
TEST_F(MeshTest, GridPointAccess) {
    Mesh mesh(*device_, MeshType::Structured);
    
    auto grid_x = mesh.get_grid_points_x();
    auto grid_y = mesh.get_grid_points_y();
    
    EXPECT_EQ(grid_x.size(), grid_y.size());
    EXPECT_GT(grid_x.size(), 0);
    
    // Check bounds
    auto extents = device_->get_extents();
    for (double x : grid_x) {
        EXPECT_GE(x, 0.0);
        EXPECT_LE(x, extents[0]);
    }
    for (double y : grid_y) {
        EXPECT_GE(y, 0.0);
        EXPECT_LE(y, extents[1]);
    }
}

// Test C interface
TEST_F(MeshTest, CInterface) {
    // Test creation
    Mesh* mesh = create_mesh(device_.get(), static_cast<int>(MeshType::Structured));
    ASSERT_NE(mesh, nullptr);
    
    // Test basic properties
    int num_nodes = mesh_get_num_nodes(mesh);
    int num_elements = mesh_get_num_elements(mesh);
    EXPECT_GT(num_nodes, 0);
    EXPECT_GT(num_elements, 0);
    
    // Test grid point access
    std::vector<double> grid_x(num_nodes), grid_y(num_nodes);
    mesh_get_grid_points_x(mesh, grid_x.data(), num_nodes);
    mesh_get_grid_points_y(mesh, grid_y.data(), num_nodes);
    
    // Check that we got valid data
    for (int i = 0; i < num_nodes; ++i) {
        EXPECT_GE(grid_x[i], 0.0);
        EXPECT_GE(grid_y[i], 0.0);
    }
    
    // Test element access
    std::vector<int> elements(num_elements * 3);
    mesh_get_elements(mesh, elements.data(), num_elements, 3);
    
    // Check element connectivity
    for (int e = 0; e < num_elements; ++e) {
        for (int n = 0; n < 3; ++n) {
            int node_id = elements[e * 3 + n];
            EXPECT_GE(node_id, 0);
            EXPECT_LT(node_id, num_nodes);
        }
    }
    
    // Test null pointer handling
    EXPECT_EQ(mesh_get_num_nodes(nullptr), 0);
    EXPECT_EQ(mesh_get_num_elements(nullptr), 0);
    mesh_get_grid_points_x(nullptr, grid_x.data(), num_nodes);
    mesh_get_grid_points_x(mesh, nullptr, num_nodes);
    mesh_get_grid_points_x(mesh, grid_x.data(), 0);
    
    // Clean up
    destroy_mesh(mesh);
    destroy_mesh(nullptr); // Should not crash
}

// Test invalid mesh creation through C interface
TEST_F(MeshTest, CInterfaceInvalidCreation) {
    Mesh* mesh = create_mesh(nullptr, static_cast<int>(MeshType::Structured));
    EXPECT_EQ(mesh, nullptr);
}

// Test copy and move semantics
TEST_F(MeshTest, CopyMoveSemantics) {
    Mesh original(*device_, MeshType::Structured);
    
    // Copy constructor
    Mesh copied(original);
    EXPECT_TRUE(copied.is_valid());
    EXPECT_EQ(copied.get_num_nodes(), original.get_num_nodes());
    EXPECT_EQ(copied.get_num_elements(), original.get_num_elements());
    
    // Copy assignment
    Device another_device(2e-6, 1e-6);
    Mesh assigned(another_device, MeshType::Structured);
    assigned = original;
    EXPECT_EQ(assigned.get_num_nodes(), original.get_num_nodes());
    EXPECT_EQ(assigned.get_num_elements(), original.get_num_elements());
    
    // Move constructor
    size_t original_nodes = original.get_num_nodes();
    size_t original_elements = original.get_num_elements();
    Mesh moved(std::move(original));
    EXPECT_TRUE(moved.is_valid());
    EXPECT_EQ(moved.get_num_nodes(), original_nodes);
    EXPECT_EQ(moved.get_num_elements(), original_elements);
}

// Performance test
TEST_F(MeshTest, PerformanceTest) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Mesh mesh(*device_, MeshType::Structured);
    
    // Perform multiple refinements
    for (int iter = 0; iter < 3; ++iter) {
        size_t num_elements = mesh.get_num_elements();
        std::vector<bool> refine_flags(num_elements, false);
        
        // Refine every 4th element
        for (size_t i = 0; i < num_elements; i += 4) {
            refine_flags[i] = true;
        }
        
        mesh.refine(refine_flags);
        EXPECT_TRUE(mesh.is_valid());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time (less than 1 second)
    EXPECT_LT(duration.count(), 1000);
}
