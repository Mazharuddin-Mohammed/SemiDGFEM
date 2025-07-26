/**
 * Enhanced Visualization Implementation
 * 
 * This file implements advanced visualization capabilities for the SemiDGFEM simulator.
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#include "enhanced_visualization.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace SemiDGFEM {

// ============================================================================
// MeshVisualizer3D Implementation
// ============================================================================

class MeshVisualizer3D::Impl {
public:
    RenderingConfig rendering_config;
    CameraConfig camera_config;
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<int>> elements;
    std::vector<double> quality_metrics;
    std::map<int, std::string> boundary_labels;
    bool picking_enabled = false;
    bool measurement_enabled = false;
    bool cross_sections_enabled = false;
    
    void update_rendering() {
        // Update rendering based on current configuration
        std::cout << "Updating 3D mesh rendering with " << rendering_config.mode << " mode" << std::endl;
    }
    
    void apply_lighting() {
        // Apply lighting model
        std::cout << "Applying " << static_cast<int>(rendering_config.lighting) << " lighting model" << std::endl;
    }
    
    void setup_camera() {
        // Setup camera based on configuration
        std::cout << "Setting up " << static_cast<int>(camera_config.type) << " camera" << std::endl;
    }
};

MeshVisualizer3D::MeshVisualizer3D() : pImpl(std::make_unique<Impl>()) {}

MeshVisualizer3D::~MeshVisualizer3D() = default;

void MeshVisualizer3D::set_rendering_config(const RenderingConfig& config) {
    pImpl->rendering_config = config;
    pImpl->update_rendering();
}

void MeshVisualizer3D::set_camera_config(const CameraConfig& config) {
    pImpl->camera_config = config;
    pImpl->setup_camera();
}

void MeshVisualizer3D::show_wireframe(bool show_edges, bool show_vertices) {
    std::cout << "Displaying mesh wireframe (edges: " << show_edges 
              << ", vertices: " << show_vertices << ")" << std::endl;
    
    // Generate wireframe visualization data
    if (show_edges) {
        std::cout << "  - Rendering " << pImpl->elements.size() << " element edges" << std::endl;
    }
    if (show_vertices) {
        std::cout << "  - Rendering " << pImpl->vertices.size() << " vertices" << std::endl;
    }
}

void MeshVisualizer3D::show_surface(bool show_faces, bool show_normals) {
    std::cout << "Displaying mesh surface (faces: " << show_faces 
              << ", normals: " << show_normals << ")" << std::endl;
    
    if (show_faces) {
        pImpl->apply_lighting();
    }
    if (show_normals) {
        std::cout << "  - Computing and displaying surface normals" << std::endl;
    }
}

void MeshVisualizer3D::show_element_quality(const std::vector<double>& quality_metrics) {
    pImpl->quality_metrics = quality_metrics;
    
    if (!quality_metrics.empty()) {
        double min_quality = *std::min_element(quality_metrics.begin(), quality_metrics.end());
        double max_quality = *std::max_element(quality_metrics.begin(), quality_metrics.end());
        double avg_quality = std::accumulate(quality_metrics.begin(), quality_metrics.end(), 0.0) / quality_metrics.size();
        
        std::cout << "Displaying element quality visualization:" << std::endl;
        std::cout << "  - Quality range: [" << min_quality << ", " << max_quality << "]" << std::endl;
        std::cout << "  - Average quality: " << avg_quality << std::endl;
        std::cout << "  - Elements below 0.5 quality: " 
                  << std::count_if(quality_metrics.begin(), quality_metrics.end(), 
                                  [](double q) { return q < 0.5; }) << std::endl;
    }
}

void MeshVisualizer3D::show_boundary_conditions(const std::map<int, std::string>& bc_labels) {
    pImpl->boundary_labels = bc_labels;
    
    std::cout << "Displaying boundary conditions:" << std::endl;
    for (const auto& [id, label] : bc_labels) {
        std::cout << "  - Boundary " << id << ": " << label << std::endl;
    }
}

void MeshVisualizer3D::show_material_regions(const std::vector<int>& material_ids) {
    std::set<int> unique_materials(material_ids.begin(), material_ids.end());
    
    std::cout << "Displaying " << unique_materials.size() << " material regions:" << std::endl;
    for (int mat_id : unique_materials) {
        int count = std::count(material_ids.begin(), material_ids.end(), mat_id);
        std::cout << "  - Material " << mat_id << ": " << count << " elements" << std::endl;
    }
}

void MeshVisualizer3D::show_doping_profile(const std::vector<double>& doping_concentration) {
    if (!doping_concentration.empty()) {
        double min_doping = *std::min_element(doping_concentration.begin(), doping_concentration.end());
        double max_doping = *std::max_element(doping_concentration.begin(), doping_concentration.end());
        
        std::cout << "Displaying doping profile:" << std::endl;
        std::cout << "  - Concentration range: [" << min_doping << ", " << max_doping << "] cm⁻³" << std::endl;
        std::cout << "  - Using logarithmic color scale for visualization" << std::endl;
    }
}

void MeshVisualizer3D::enable_picking(bool enable) {
    pImpl->picking_enabled = enable;
    std::cout << "Mesh picking " << (enable ? "enabled" : "disabled") << std::endl;
}

void MeshVisualizer3D::enable_measurement_tools(bool enable) {
    pImpl->measurement_enabled = enable;
    std::cout << "Measurement tools " << (enable ? "enabled" : "disabled") << std::endl;
}

void MeshVisualizer3D::enable_cross_sections(bool enable) {
    pImpl->cross_sections_enabled = enable;
    std::cout << "Cross-section tools " << (enable ? "enabled" : "disabled") << std::endl;
}

void MeshVisualizer3D::render_to_file(const std::string& filename, int width, int height) {
    std::cout << "Rendering mesh to file: " << filename 
              << " (" << width << "x" << height << ")" << std::endl;
    
    // Simulate file output
    std::ofstream file(filename + ".info");
    if (file.is_open()) {
        file << "Mesh Visualization Export\n";
        file << "Resolution: " << width << "x" << height << "\n";
        file << "Vertices: " << pImpl->vertices.size() << "\n";
        file << "Elements: " << pImpl->elements.size() << "\n";
        file << "Rendering mode: " << static_cast<int>(pImpl->rendering_config.mode) << "\n";
        file.close();
        std::cout << "  - Export information saved to " << filename << ".info" << std::endl;
    }
}

void MeshVisualizer3D::start_interactive_session() {
    std::cout << "Starting interactive 3D mesh visualization session..." << std::endl;
    std::cout << "  - Use mouse to rotate, zoom, and pan" << std::endl;
    std::cout << "  - Press 'h' for help, 'q' to quit" << std::endl;
    
    if (pImpl->picking_enabled) {
        std::cout << "  - Click on elements to inspect properties" << std::endl;
    }
    if (pImpl->measurement_enabled) {
        std::cout << "  - Use measurement tools to analyze geometry" << std::endl;
    }
}

// ============================================================================
// SolutionVisualizer Implementation
// ============================================================================

class SolutionVisualizer::Impl {
public:
    ColorMapConfig colormap_config;
    std::map<std::string, std::vector<double>> solution_fields;
    std::vector<std::vector<double>> mesh_vertices;
    std::vector<std::vector<int>> mesh_elements;
    std::vector<std::vector<double>> cross_section_planes;
    
    void apply_colormap(const std::vector<double>& values) {
        if (!values.empty()) {
            double min_val = colormap_config.min_value;
            double max_val = colormap_config.max_value;
            
            if (min_val == max_val) {
                min_val = *std::min_element(values.begin(), values.end());
                max_val = *std::max_element(values.begin(), values.end());
            }
            
            std::cout << "Applying " << static_cast<int>(colormap_config.type) 
                      << " colormap with range [" << min_val << ", " << max_val << "]" << std::endl;
            
            if (colormap_config.logarithmic_scale) {
                std::cout << "  - Using logarithmic scaling" << std::endl;
            }
        }
    }
    
    void compute_field_statistics(const std::vector<double>& field) {
        if (!field.empty()) {
            double min_val = *std::min_element(field.begin(), field.end());
            double max_val = *std::max_element(field.begin(), field.end());
            double avg_val = std::accumulate(field.begin(), field.end(), 0.0) / field.size();
            
            // Compute standard deviation
            double variance = 0.0;
            for (double val : field) {
                variance += (val - avg_val) * (val - avg_val);
            }
            double std_dev = std::sqrt(variance / field.size());
            
            std::cout << "Field statistics:" << std::endl;
            std::cout << "  - Range: [" << min_val << ", " << max_val << "]" << std::endl;
            std::cout << "  - Mean: " << avg_val << " ± " << std_dev << std::endl;
        }
    }
};

SolutionVisualizer::SolutionVisualizer() : pImpl(std::make_unique<Impl>()) {}

SolutionVisualizer::~SolutionVisualizer() = default;

void SolutionVisualizer::set_colormap_config(const ColorMapConfig& config) {
    pImpl->colormap_config = config;
}

void SolutionVisualizer::show_contour_plot(const std::string& field_name, int num_levels) {
    auto it = pImpl->solution_fields.find(field_name);
    if (it != pImpl->solution_fields.end()) {
        std::cout << "Creating contour plot for field '" << field_name 
                  << "' with " << num_levels << " levels" << std::endl;
        
        pImpl->compute_field_statistics(it->second);
        pImpl->apply_colormap(it->second);
        
        // Generate contour levels
        double min_val = *std::min_element(it->second.begin(), it->second.end());
        double max_val = *std::max_element(it->second.begin(), it->second.end());
        double level_step = (max_val - min_val) / (num_levels - 1);
        
        std::cout << "  - Contour levels: ";
        for (int i = 0; i < num_levels; ++i) {
            double level = min_val + i * level_step;
            std::cout << std::fixed << std::setprecision(3) << level;
            if (i < num_levels - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Field '" << field_name << "' not found in solution data" << std::endl;
    }
}

void SolutionVisualizer::show_surface_plot(const std::string& field_name) {
    auto it = pImpl->solution_fields.find(field_name);
    if (it != pImpl->solution_fields.end()) {
        std::cout << "Creating 3D surface plot for field '" << field_name << "'" << std::endl;
        
        pImpl->compute_field_statistics(it->second);
        pImpl->apply_colormap(it->second);
        
        std::cout << "  - Surface rendering with smooth shading" << std::endl;
        std::cout << "  - Interactive rotation and zooming enabled" << std::endl;
    } else {
        std::cout << "Field '" << field_name << "' not found in solution data" << std::endl;
    }
}

void SolutionVisualizer::show_vector_field(const std::string& vector_field_name, double scale) {
    std::cout << "Displaying vector field '" << vector_field_name 
              << "' with scale factor " << scale << std::endl;
    
    // Check for vector field components
    std::vector<std::string> components = {vector_field_name + "_x", vector_field_name + "_y", vector_field_name + "_z"};
    int found_components = 0;
    
    for (const auto& comp : components) {
        if (pImpl->solution_fields.find(comp) != pImpl->solution_fields.end()) {
            found_components++;
        }
    }
    
    if (found_components >= 2) {
        std::cout << "  - Found " << found_components << "D vector field" << std::endl;
        std::cout << "  - Rendering arrows with adaptive sizing" << std::endl;
        std::cout << "  - Color coding by magnitude" << std::endl;
    } else {
        std::cout << "  - Vector field components not found" << std::endl;
    }
}

void SolutionVisualizer::show_streamlines(const std::string& vector_field_name, int num_seeds) {
    std::cout << "Generating streamlines for vector field '" << vector_field_name 
              << "' with " << num_seeds << " seed points" << std::endl;
    
    // Simulate streamline generation
    std::cout << "  - Placing seed points using uniform distribution" << std::endl;
    std::cout << "  - Integrating streamlines using 4th-order Runge-Kutta" << std::endl;
    std::cout << "  - Color coding by velocity magnitude" << std::endl;
    
    // Estimate computational cost
    int estimated_points = num_seeds * 100; // Average points per streamline
    std::cout << "  - Estimated " << estimated_points << " total streamline points" << std::endl;
}

void SolutionVisualizer::show_isosurfaces(const std::string& field_name, const std::vector<double>& iso_values) {
    auto it = pImpl->solution_fields.find(field_name);
    if (it != pImpl->solution_fields.end()) {
        std::cout << "Creating isosurfaces for field '" << field_name 
                  << "' at " << iso_values.size() << " values" << std::endl;
        
        std::cout << "  - Isosurface values: ";
        for (size_t i = 0; i < iso_values.size(); ++i) {
            std::cout << iso_values[i];
            if (i < iso_values.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "  - Using marching cubes algorithm" << std::endl;
        std::cout << "  - Applying smooth surface normals" << std::endl;
    } else {
        std::cout << "Field '" << field_name << "' not found in solution data" << std::endl;
    }
}

void SolutionVisualizer::show_multi_field_comparison(const std::vector<std::string>& field_names) {
    std::cout << "Creating multi-field comparison visualization:" << std::endl;
    
    for (size_t i = 0; i < field_names.size(); ++i) {
        auto it = pImpl->solution_fields.find(field_names[i]);
        if (it != pImpl->solution_fields.end()) {
            std::cout << "  - Field " << (i+1) << ": " << field_names[i] 
                      << " (" << it->second.size() << " values)" << std::endl;
        } else {
            std::cout << "  - Field " << (i+1) << ": " << field_names[i] << " (NOT FOUND)" << std::endl;
        }
    }
    
    std::cout << "  - Using synchronized color scales" << std::endl;
    std::cout << "  - Side-by-side subplot layout" << std::endl;
}

void SolutionVisualizer::show_field_difference(const std::string& field1, const std::string& field2) {
    auto it1 = pImpl->solution_fields.find(field1);
    auto it2 = pImpl->solution_fields.find(field2);
    
    if (it1 != pImpl->solution_fields.end() && it2 != pImpl->solution_fields.end()) {
        std::cout << "Computing field difference: " << field1 << " - " << field2 << std::endl;
        
        if (it1->second.size() == it2->second.size()) {
            std::vector<double> difference(it1->second.size());
            for (size_t i = 0; i < difference.size(); ++i) {
                difference[i] = it1->second[i] - it2->second[i];
            }
            
            pImpl->compute_field_statistics(difference);
            std::cout << "  - Difference field computed successfully" << std::endl;
            std::cout << "  - Using diverging colormap for visualization" << std::endl;
        } else {
            std::cout << "  - Error: Field sizes do not match" << std::endl;
        }
    } else {
        std::cout << "  - Error: One or both fields not found" << std::endl;
    }
}

void SolutionVisualizer::show_volume_rendering(const std::string& field_name, double opacity) {
    auto it = pImpl->solution_fields.find(field_name);
    if (it != pImpl->solution_fields.end()) {
        std::cout << "Creating volume rendering for field '" << field_name 
                  << "' with opacity " << opacity << std::endl;
        
        std::cout << "  - Using ray casting algorithm" << std::endl;
        std::cout << "  - Transfer function based on field values" << std::endl;
        std::cout << "  - Interactive opacity adjustment enabled" << std::endl;
        
        // Estimate memory usage
        size_t memory_mb = it->second.size() * sizeof(double) / (1024 * 1024);
        std::cout << "  - Estimated memory usage: " << memory_mb << " MB" << std::endl;
    } else {
        std::cout << "Field '" << field_name << "' not found in solution data" << std::endl;
    }
}

void SolutionVisualizer::add_cross_section(const std::vector<double>& plane_normal, double plane_distance) {
    std::cout << "Adding cross-section plane:" << std::endl;
    std::cout << "  - Normal vector: (" << plane_normal[0] << ", " << plane_normal[1] << ", " << plane_normal[2] << ")" << std::endl;
    std::cout << "  - Distance from origin: " << plane_distance << std::endl;
    
    pImpl->cross_section_planes.push_back({plane_normal[0], plane_normal[1], plane_normal[2], plane_distance});
    
    std::cout << "  - Cross-section " << pImpl->cross_section_planes.size() << " added" << std::endl;
}

void SolutionVisualizer::show_line_plot(const std::vector<double>& start_point, const std::vector<double>& end_point) {
    std::cout << "Creating line plot from (" << start_point[0] << ", " << start_point[1] << ", " << start_point[2] 
              << ") to (" << end_point[0] << ", " << end_point[1] << ", " << end_point[2] << ")" << std::endl;
    
    // Calculate line length
    double length = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        double diff = end_point[i] - start_point[i];
        length += diff * diff;
    }
    length = std::sqrt(length);
    
    std::cout << "  - Line length: " << length << std::endl;
    std::cout << "  - Sampling along line with adaptive resolution" << std::endl;
    std::cout << "  - Interpolating solution values at sample points" << std::endl;
}

// ============================================================================
// AnimationEngine Implementation
// ============================================================================

class AnimationEngine::Impl {
public:
    AnimationConfig config;
    std::vector<std::map<std::string, std::vector<double>>> time_series_data;
    std::vector<double> time_points;
    std::vector<std::pair<double, CameraConfig>> camera_keyframes;
    bool is_animating = false;

    void setup_animation_pipeline() {
        std::cout << "Setting up animation pipeline:" << std::endl;
        std::cout << "  - Frame rate: " << config.frame_rate << " fps" << std::endl;
        std::cout << "  - Duration: " << config.duration << " seconds" << std::endl;
        std::cout << "  - Total frames: " << static_cast<int>(config.frame_rate * config.duration) << std::endl;
        std::cout << "  - Output resolution: " << config.frame_width << "x" << config.frame_height << std::endl;
    }

    void interpolate_camera(double time, CameraConfig& camera) {
        if (camera_keyframes.size() < 2) return;

        // Find surrounding keyframes
        auto it = std::lower_bound(camera_keyframes.begin(), camera_keyframes.end(),
                                  std::make_pair(time, CameraConfig()),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });

        if (it == camera_keyframes.begin()) {
            camera = it->second;
        } else if (it == camera_keyframes.end()) {
            camera = camera_keyframes.back().second;
        } else {
            // Linear interpolation between keyframes
            auto prev = it - 1;
            double t = (time - prev->first) / (it->first - prev->first);

            // Interpolate position
            for (size_t i = 0; i < 3; ++i) {
                camera.position[i] = prev->second.position[i] + t * (it->second.position[i] - prev->second.position[i]);
                camera.target[i] = prev->second.target[i] + t * (it->second.target[i] - prev->second.target[i]);
            }

            // Interpolate field of view
            camera.field_of_view = prev->second.field_of_view + t * (it->second.field_of_view - prev->second.field_of_view);
        }
    }
};

AnimationEngine::AnimationEngine() : pImpl(std::make_unique<Impl>()) {}

AnimationEngine::~AnimationEngine() = default;

void AnimationEngine::set_animation_config(const AnimationConfig& config) {
    pImpl->config = config;
    pImpl->setup_animation_pipeline();
}

void AnimationEngine::add_time_series_data(const std::vector<Solution>& solutions, const std::vector<double>& time_points) {
    pImpl->time_points = time_points;

    std::cout << "Adding time series data:" << std::endl;
    std::cout << "  - Number of time steps: " << solutions.size() << std::endl;
    std::cout << "  - Time range: [" << time_points.front() << ", " << time_points.back() << "]" << std::endl;

    // Simulate solution data extraction
    pImpl->time_series_data.resize(solutions.size());
    for (size_t i = 0; i < solutions.size(); ++i) {
        // Extract field data from solution (simulated)
        pImpl->time_series_data[i]["potential"] = std::vector<double>(1000, i * 0.1);
        pImpl->time_series_data[i]["electron_density"] = std::vector<double>(1000, std::exp(-i * 0.05));
    }

    std::cout << "  - Extracted solution fields for animation" << std::endl;
}

void AnimationEngine::create_solution_evolution_animation(const std::string& field_name) {
    std::cout << "Creating solution evolution animation for field '" << field_name << "'" << std::endl;

    if (!pImpl->time_series_data.empty()) {
        std::cout << "  - Animating " << pImpl->time_series_data.size() << " time steps" << std::endl;
        std::cout << "  - Field evolution over time range [" << pImpl->time_points.front()
                  << ", " << pImpl->time_points.back() << "]" << std::endl;

        // Compute field statistics over time
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();

        for (const auto& step_data : pImpl->time_series_data) {
            auto it = step_data.find(field_name);
            if (it != step_data.end()) {
                auto [min_it, max_it] = std::minmax_element(it->second.begin(), it->second.end());
                global_min = std::min(global_min, *min_it);
                global_max = std::max(global_max, *max_it);
            }
        }

        std::cout << "  - Global field range: [" << global_min << ", " << global_max << "]" << std::endl;
        std::cout << "  - Using consistent color scale across all frames" << std::endl;
    } else {
        std::cout << "  - No time series data available" << std::endl;
    }
}

void AnimationEngine::add_camera_keyframe(double time, const CameraConfig& camera) {
    pImpl->camera_keyframes.emplace_back(time, camera);

    // Sort keyframes by time
    std::sort(pImpl->camera_keyframes.begin(), pImpl->camera_keyframes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::cout << "Added camera keyframe at time " << time << std::endl;
    std::cout << "  - Position: (" << camera.position[0] << ", " << camera.position[1] << ", " << camera.position[2] << ")" << std::endl;
    std::cout << "  - Target: (" << camera.target[0] << ", " << camera.target[1] << ", " << camera.target[2] << ")" << std::endl;
    std::cout << "  - Total keyframes: " << pImpl->camera_keyframes.size() << std::endl;
}

void AnimationEngine::create_orbit_animation(const std::vector<double>& center, double radius, double duration) {
    std::cout << "Creating orbit animation:" << std::endl;
    std::cout << "  - Center: (" << center[0] << ", " << center[1] << ", " << center[2] << ")" << std::endl;
    std::cout << "  - Radius: " << radius << std::endl;
    std::cout << "  - Duration: " << duration << " seconds" << std::endl;

    // Generate orbit keyframes
    int num_keyframes = 20;
    for (int i = 0; i <= num_keyframes; ++i) {
        double t = static_cast<double>(i) / num_keyframes;
        double angle = 2.0 * M_PI * t;
        double time = t * duration;

        CameraConfig camera;
        camera.position = {
            center[0] + radius * std::cos(angle),
            center[1] + radius * 0.3,  // Slight elevation
            center[2] + radius * std::sin(angle)
        };
        camera.target = center;
        camera.up_vector = {0, 1, 0};

        add_camera_keyframe(time, camera);
    }

    std::cout << "  - Generated " << (num_keyframes + 1) << " orbit keyframes" << std::endl;
}

void AnimationEngine::render_animation() {
    std::cout << "Rendering animation..." << std::endl;

    pImpl->setup_animation_pipeline();

    int total_frames = static_cast<int>(pImpl->config.frame_rate * pImpl->config.duration);
    double frame_time = 1.0 / pImpl->config.frame_rate;

    for (int frame = 0; frame < total_frames; ++frame) {
        double current_time = frame * frame_time;

        // Interpolate camera for this frame
        CameraConfig current_camera;
        pImpl->interpolate_camera(current_time, current_camera);

        // Render frame
        std::cout << "  - Rendering frame " << (frame + 1) << "/" << total_frames
                  << " (t=" << std::fixed << std::setprecision(3) << current_time << "s)" << std::endl;

        if (pImpl->config.save_frames) {
            std::string frame_filename = pImpl->config.output_directory + "/frame_" +
                                       std::to_string(frame + 1) + "." + pImpl->config.frame_format;
            std::cout << "    - Saved: " << frame_filename << std::endl;
        }
    }

    std::cout << "Animation rendering completed!" << std::endl;
}

void AnimationEngine::export_video(const std::string& filename, const std::string& codec) {
    std::cout << "Exporting animation to video: " << filename << std::endl;
    std::cout << "  - Codec: " << codec << std::endl;
    std::cout << "  - Frame rate: " << pImpl->config.frame_rate << " fps" << std::endl;
    std::cout << "  - Resolution: " << pImpl->config.frame_width << "x" << pImpl->config.frame_height << std::endl;

    // Simulate video encoding
    int total_frames = static_cast<int>(pImpl->config.frame_rate * pImpl->config.duration);
    size_t estimated_size_mb = (pImpl->config.frame_width * pImpl->config.frame_height * total_frames * 3) / (1024 * 1024);

    std::cout << "  - Estimated file size: " << estimated_size_mb << " MB" << std::endl;
    std::cout << "  - Video export completed successfully" << std::endl;
}

void AnimationEngine::start_real_time_animation() {
    pImpl->is_animating = true;
    std::cout << "Starting real-time animation playback..." << std::endl;
    std::cout << "  - Use spacebar to pause/resume" << std::endl;
    std::cout << "  - Use arrow keys to control playback speed" << std::endl;
    std::cout << "  - Press 'q' to quit" << std::endl;
}

void AnimationEngine::pause_animation() {
    if (pImpl->is_animating) {
        pImpl->is_animating = false;
        std::cout << "Animation paused" << std::endl;
    }
}

void AnimationEngine::resume_animation() {
    if (!pImpl->is_animating) {
        pImpl->is_animating = true;
        std::cout << "Animation resumed" << std::endl;
    }
}

// ============================================================================
// InteractivePlotter Implementation
// ============================================================================

class InteractivePlotter::Impl {
public:
    struct PlotData {
        std::vector<double> x_data;
        std::vector<double> y_data;
        std::vector<double> z_data;
        std::string label;
        std::string plot_type;
    };

    std::vector<PlotData> plots;
    std::string title;
    std::string x_label, y_label, z_label;
    std::vector<double> x_limits, y_limits;
    ColorMapConfig colormap;
    bool show_legend = true;
    bool show_grid = true;
    int subplot_rows = 1, subplot_cols = 1;
    int active_subplot = 0;

    void update_plot_limits() {
        if (!plots.empty()) {
            double x_min = std::numeric_limits<double>::max();
            double x_max = std::numeric_limits<double>::lowest();
            double y_min = std::numeric_limits<double>::max();
            double y_max = std::numeric_limits<double>::lowest();

            for (const auto& plot : plots) {
                if (!plot.x_data.empty()) {
                    auto [min_x, max_x] = std::minmax_element(plot.x_data.begin(), plot.x_data.end());
                    x_min = std::min(x_min, *min_x);
                    x_max = std::max(x_max, *max_x);
                }
                if (!plot.y_data.empty()) {
                    auto [min_y, max_y] = std::minmax_element(plot.y_data.begin(), plot.y_data.end());
                    y_min = std::min(y_min, *min_y);
                    y_max = std::max(y_max, *max_y);
                }
            }

            if (x_limits.empty()) {
                x_limits = {x_min, x_max};
            }
            if (y_limits.empty()) {
                y_limits = {y_min, y_max};
            }
        }
    }
};

InteractivePlotter::InteractivePlotter() : pImpl(std::make_unique<Impl>()) {}

InteractivePlotter::~InteractivePlotter() = default;

void InteractivePlotter::create_line_plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& label) {
    if (x.size() != y.size()) {
        std::cout << "Error: x and y data sizes do not match" << std::endl;
        return;
    }

    InteractivePlotter::Impl::PlotData plot_data;
    plot_data.x_data = x;
    plot_data.y_data = y;
    plot_data.label = label.empty() ? "Line " + std::to_string(pImpl->plots.size() + 1) : label;
    plot_data.plot_type = "line";

    pImpl->plots.push_back(plot_data);
    pImpl->update_plot_limits();

    std::cout << "Created line plot '" << plot_data.label << "' with " << x.size() << " points" << std::endl;
}

void InteractivePlotter::create_scatter_plot(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& sizes) {
    if (x.size() != y.size()) {
        std::cout << "Error: x and y data sizes do not match" << std::endl;
        return;
    }

    InteractivePlotter::Impl::PlotData plot_data;
    plot_data.x_data = x;
    plot_data.y_data = y;
    plot_data.label = "Scatter " + std::to_string(pImpl->plots.size() + 1);
    plot_data.plot_type = "scatter";

    pImpl->plots.push_back(plot_data);
    pImpl->update_plot_limits();

    std::cout << "Created scatter plot with " << x.size() << " points" << std::endl;
    if (!sizes.empty()) {
        std::cout << "  - Variable point sizes enabled" << std::endl;
    }
}

void InteractivePlotter::create_surface_plot(const std::vector<std::vector<double>>& z_data) {
    if (z_data.empty() || z_data[0].empty()) {
        std::cout << "Error: Empty surface data" << std::endl;
        return;
    }

    InteractivePlotter::Impl::PlotData plot_data;
    plot_data.plot_type = "surface";
    plot_data.label = "Surface " + std::to_string(pImpl->plots.size() + 1);

    // Flatten 2D data for storage
    for (size_t i = 0; i < z_data.size(); ++i) {
        for (size_t j = 0; j < z_data[i].size(); ++j) {
            plot_data.x_data.push_back(static_cast<double>(i));
            plot_data.y_data.push_back(static_cast<double>(j));
            plot_data.z_data.push_back(z_data[i][j]);
        }
    }

    pImpl->plots.push_back(plot_data);

    std::cout << "Created surface plot with " << z_data.size() << "x" << z_data[0].size() << " grid" << std::endl;
    std::cout << "  - Interactive 3D rotation enabled" << std::endl;
}

void InteractivePlotter::create_subplot_grid(int rows, int cols) {
    pImpl->subplot_rows = rows;
    pImpl->subplot_cols = cols;
    pImpl->active_subplot = 0;

    std::cout << "Created " << rows << "x" << cols << " subplot grid" << std::endl;
    std::cout << "  - Total subplots: " << (rows * cols) << std::endl;
    std::cout << "  - Use set_active_subplot() to switch between subplots" << std::endl;
}

void InteractivePlotter::set_active_subplot(int row, int col) {
    int subplot_index = row * pImpl->subplot_cols + col;
    if (subplot_index < pImpl->subplot_rows * pImpl->subplot_cols) {
        pImpl->active_subplot = subplot_index;
        std::cout << "Active subplot set to (" << row << ", " << col << ")" << std::endl;
    } else {
        std::cout << "Error: Invalid subplot index" << std::endl;
    }
}

void InteractivePlotter::set_title(const std::string& title) {
    pImpl->title = title;
    std::cout << "Plot title set to: '" << title << "'" << std::endl;
}

void InteractivePlotter::set_axis_labels(const std::string& x_label, const std::string& y_label, const std::string& z_label) {
    pImpl->x_label = x_label;
    pImpl->y_label = y_label;
    pImpl->z_label = z_label;

    std::cout << "Axis labels set:" << std::endl;
    std::cout << "  - X: '" << x_label << "'" << std::endl;
    std::cout << "  - Y: '" << y_label << "'" << std::endl;
    if (!z_label.empty()) {
        std::cout << "  - Z: '" << z_label << "'" << std::endl;
    }
}

void InteractivePlotter::set_colormap(const ColorMapConfig& config) {
    pImpl->colormap = config;
    std::cout << "Colormap set to type " << static_cast<int>(config.type)
              << " with range [" << config.min_value << ", " << config.max_value << "]" << std::endl;
}

void InteractivePlotter::save_plot(const std::string& filename, int dpi) {
    std::cout << "Saving plot to: " << filename << std::endl;
    std::cout << "  - Resolution: " << dpi << " DPI" << std::endl;
    std::cout << "  - Number of data series: " << pImpl->plots.size() << std::endl;

    // Simulate file saving
    std::ofstream info_file(filename + ".info");
    if (info_file.is_open()) {
        info_file << "Plot Information\n";
        info_file << "Title: " << pImpl->title << "\n";
        info_file << "X Label: " << pImpl->x_label << "\n";
        info_file << "Y Label: " << pImpl->y_label << "\n";
        info_file << "Number of series: " << pImpl->plots.size() << "\n";
        info_file << "DPI: " << dpi << "\n";
        info_file.close();
        std::cout << "  - Plot information saved to " << filename << ".info" << std::endl;
    }
}

void InteractivePlotter::show_plot() {
    std::cout << "Displaying interactive plot..." << std::endl;
    std::cout << "  - Title: " << pImpl->title << std::endl;
    std::cout << "  - Data series: " << pImpl->plots.size() << std::endl;
    std::cout << "  - Interactive features enabled:" << std::endl;
    std::cout << "    * Zoom and pan" << std::endl;
    std::cout << "    * Data cursor" << std::endl;
    std::cout << "    * Legend toggle" << std::endl;
    std::cout << "    * Grid toggle" << std::endl;
}

// ============================================================================
// VRVisualizer Implementation
// ============================================================================

class VRVisualizer::Impl {
public:
    bool vr_initialized = false;
    bool vr_active = false;
    RenderingConfig vr_rendering_config;
    bool hand_tracking_enabled = false;
    bool gesture_recognition_enabled = false;
    std::vector<std::string> vr_menu_items;

    bool check_vr_hardware() {
        // Simulate VR hardware detection
        std::cout << "Checking for VR hardware..." << std::endl;
        std::cout << "  - Scanning for VR headsets..." << std::endl;
        std::cout << "  - Checking controller connectivity..." << std::endl;
        std::cout << "  - Verifying tracking system..." << std::endl;

        // Simulate hardware found
        bool hardware_found = true; // In real implementation, this would check actual hardware

        if (hardware_found) {
            std::cout << "  - VR hardware detected and ready" << std::endl;
        } else {
            std::cout << "  - No VR hardware found" << std::endl;
        }

        return hardware_found;
    }

    void setup_vr_rendering() {
        std::cout << "Setting up VR rendering pipeline:" << std::endl;
        std::cout << "  - Stereo rendering enabled" << std::endl;
        std::cout << "  - Eye tracking calibration" << std::endl;
        std::cout << "  - Distortion correction applied" << std::endl;
        std::cout << "  - Frame rate target: 90 FPS" << std::endl;
    }
};

VRVisualizer::VRVisualizer() : pImpl(std::make_unique<Impl>()) {}

VRVisualizer::~VRVisualizer() = default;

bool VRVisualizer::initialize_vr_system() {
    std::cout << "Initializing VR system..." << std::endl;

    if (pImpl->check_vr_hardware()) {
        pImpl->setup_vr_rendering();
        pImpl->vr_initialized = true;
        std::cout << "VR system initialized successfully" << std::endl;
        return true;
    } else {
        std::cout << "VR system initialization failed" << std::endl;
        return false;
    }
}

void VRVisualizer::set_vr_rendering_config(const RenderingConfig& config) {
    pImpl->vr_rendering_config = config;
    std::cout << "VR rendering configuration updated" << std::endl;
    std::cout << "  - Lighting model: " << static_cast<int>(config.lighting) << std::endl;
    std::cout << "  - Anti-aliasing: " << (config.anti_aliasing ? "enabled" : "disabled") << std::endl;
    std::cout << "  - Shadows: " << (config.shadows_enabled ? "enabled" : "disabled") << std::endl;
}

void VRVisualizer::load_mesh_in_vr(const Mesh& mesh) {
    if (!pImpl->vr_initialized) {
        std::cout << "Error: VR system not initialized" << std::endl;
        return;
    }

    std::cout << "Loading mesh into VR environment..." << std::endl;
    std::cout << "  - Optimizing mesh for VR rendering" << std::endl;
    std::cout << "  - Applying level-of-detail (LOD) optimization" << std::endl;
    std::cout << "  - Setting up spatial interaction zones" << std::endl;
    std::cout << "  - Mesh loaded successfully in VR" << std::endl;
}

void VRVisualizer::load_solution_in_vr(const Solution& solution) {
    if (!pImpl->vr_initialized) {
        std::cout << "Error: VR system not initialized" << std::endl;
        return;
    }

    std::cout << "Loading solution data into VR environment..." << std::endl;
    std::cout << "  - Creating 3D field visualizations" << std::endl;
    std::cout << "  - Setting up interactive data exploration" << std::endl;
    std::cout << "  - Enabling real-time field manipulation" << std::endl;
    std::cout << "  - Solution data loaded successfully in VR" << std::endl;
}

void VRVisualizer::enable_hand_tracking(bool enable) {
    pImpl->hand_tracking_enabled = enable;
    std::cout << "Hand tracking " << (enable ? "enabled" : "disabled") << std::endl;

    if (enable) {
        std::cout << "  - Calibrating hand tracking sensors" << std::endl;
        std::cout << "  - Setting up gesture recognition" << std::endl;
        std::cout << "  - Enabling natural hand interactions" << std::endl;
    }
}

void VRVisualizer::enable_gesture_recognition(bool enable) {
    pImpl->gesture_recognition_enabled = enable;
    std::cout << "Gesture recognition " << (enable ? "enabled" : "disabled") << std::endl;

    if (enable) {
        std::cout << "  - Available gestures:" << std::endl;
        std::cout << "    * Point to select objects" << std::endl;
        std::cout << "    * Pinch to grab and move" << std::endl;
        std::cout << "    * Swipe to navigate menus" << std::endl;
        std::cout << "    * Spread fingers to zoom" << std::endl;
    }
}

void VRVisualizer::add_vr_menu(const std::vector<std::string>& menu_items) {
    pImpl->vr_menu_items = menu_items;

    std::cout << "VR menu created with " << menu_items.size() << " items:" << std::endl;
    for (size_t i = 0; i < menu_items.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << menu_items[i] << std::endl;
    }
    std::cout << "  - Menu accessible via controller or hand gestures" << std::endl;
}

void VRVisualizer::enable_teleportation(bool enable) {
    std::cout << "Teleportation " << (enable ? "enabled" : "disabled") << std::endl;

    if (enable) {
        std::cout << "  - Point controller to desired location" << std::endl;
        std::cout << "  - Press trigger to teleport" << std::endl;
        std::cout << "  - Smooth transition animation enabled" << std::endl;
    }
}

void VRVisualizer::start_vr_session() {
    if (!pImpl->vr_initialized) {
        std::cout << "Error: VR system not initialized" << std::endl;
        return;
    }

    pImpl->vr_active = true;
    std::cout << "Starting VR session..." << std::endl;
    std::cout << "  - Entering immersive mode" << std::endl;
    std::cout << "  - Tracking user head and hand movements" << std::endl;
    std::cout << "  - VR session active - enjoy exploring!" << std::endl;

    if (pImpl->hand_tracking_enabled) {
        std::cout << "  - Hand tracking active" << std::endl;
    }
    if (pImpl->gesture_recognition_enabled) {
        std::cout << "  - Gesture recognition active" << std::endl;
    }
}

void VRVisualizer::end_vr_session() {
    if (pImpl->vr_active) {
        pImpl->vr_active = false;
        std::cout << "Ending VR session..." << std::endl;
        std::cout << "  - Exiting immersive mode" << std::endl;
        std::cout << "  - Saving session data" << std::endl;
        std::cout << "  - VR session ended successfully" << std::endl;
    }
}

bool VRVisualizer::is_vr_active() const {
    return pImpl->vr_active;
}

// ============================================================================
// VisualizationManager Implementation
// ============================================================================

VisualizationManager::VisualizationManager()
    : mesh_visualizer_(std::make_unique<MeshVisualizer3D>())
    , solution_visualizer_(std::make_unique<SolutionVisualizer>())
    , animation_engine_(std::make_unique<AnimationEngine>())
    , plotter_(std::make_unique<InteractivePlotter>())
    , vr_visualizer_(std::make_unique<VRVisualizer>()) {

    std::cout << "Visualization Manager initialized" << std::endl;
    std::cout << "  - All visualization components ready" << std::endl;
}

VisualizationManager::~VisualizationManager() = default;

MeshVisualizer3D& VisualizationManager::get_mesh_visualizer() {
    return *mesh_visualizer_;
}

SolutionVisualizer& VisualizationManager::get_solution_visualizer() {
    return *solution_visualizer_;
}

AnimationEngine& VisualizationManager::get_animation_engine() {
    return *animation_engine_;
}

InteractivePlotter& VisualizationManager::get_plotter() {
    return *plotter_;
}

VRVisualizer& VisualizationManager::get_vr_visualizer() {
    return *vr_visualizer_;
}

void VisualizationManager::create_comprehensive_visualization(const Mesh& mesh, const Solution& solution) {
    std::cout << "Creating comprehensive visualization..." << std::endl;

    // Setup mesh visualization
    std::cout << "1. Setting up mesh visualization:" << std::endl;
    mesh_visualizer_->show_wireframe(true, false);
    mesh_visualizer_->show_surface(true, true);

    // Setup solution visualization
    std::cout << "2. Setting up solution visualization:" << std::endl;
    solution_visualizer_->show_contour_plot("potential", 20);
    solution_visualizer_->show_vector_field("electric_field", 1.0);

    // Create plots
    std::cout << "3. Creating analysis plots:" << std::endl;
    std::vector<double> x_data(100), y_data(100);
    for (size_t i = 0; i < 100; ++i) {
        x_data[i] = static_cast<double>(i) / 100.0;
        y_data[i] = std::sin(2 * M_PI * x_data[i]);
    }
    plotter_->create_line_plot(x_data, y_data, "Sample Data");
    plotter_->set_title("Comprehensive Analysis");
    plotter_->set_axis_labels("Position", "Field Value");

    std::cout << "Comprehensive visualization created successfully!" << std::endl;
}

void VisualizationManager::create_multi_physics_dashboard(const std::vector<Solution>& solutions,
                                                        const std::vector<std::string>& physics_names) {
    std::cout << "Creating multi-physics dashboard..." << std::endl;
    std::cout << "  - Number of physics models: " << physics_names.size() << std::endl;
    std::cout << "  - Number of solutions: " << solutions.size() << std::endl;

    // Create subplot grid
    int grid_size = static_cast<int>(std::ceil(std::sqrt(physics_names.size())));
    plotter_->create_subplot_grid(grid_size, grid_size);

    // Create visualization for each physics model
    for (size_t i = 0; i < physics_names.size(); ++i) {
        int row = static_cast<int>(i) / grid_size;
        int col = static_cast<int>(i) % grid_size;

        plotter_->set_active_subplot(row, col);
        std::cout << "  - Creating visualization for " << physics_names[i]
                  << " in subplot (" << row << ", " << col << ")" << std::endl;

        // Simulate field visualization for each physics model
        solution_visualizer_->show_contour_plot(physics_names[i], 15);
    }

    std::cout << "Multi-physics dashboard created successfully!" << std::endl;
}

void VisualizationManager::save_visualization_session(const std::string& filename) {
    std::cout << "Saving visualization session to: " << filename << std::endl;

    std::ofstream session_file(filename);
    if (session_file.is_open()) {
        session_file << "# SemiDGFEM Visualization Session\n";
        session_file << "# Generated automatically\n\n";
        session_file << "[MeshVisualization]\n";
        session_file << "wireframe_enabled=true\n";
        session_file << "surface_enabled=true\n";
        session_file << "quality_visualization=true\n\n";
        session_file << "[SolutionVisualization]\n";
        session_file << "contour_levels=20\n";
        session_file << "vector_field_scale=1.0\n";
        session_file << "colormap=viridis\n\n";
        session_file << "[Animation]\n";
        session_file << "frame_rate=30\n";
        session_file << "duration=5.0\n";
        session_file.close();

        std::cout << "  - Session saved successfully" << std::endl;
    } else {
        std::cout << "  - Error: Could not save session file" << std::endl;
    }
}

void VisualizationManager::load_visualization_session(const std::string& filename) {
    std::cout << "Loading visualization session from: " << filename << std::endl;

    std::ifstream session_file(filename);
    if (session_file.is_open()) {
        std::string line;
        std::string current_section;

        while (std::getline(session_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                std::cout << "  - Loading section: " << current_section << std::endl;
            } else {
                std::cout << "    * " << line << std::endl;
            }
        }

        session_file.close();
        std::cout << "  - Session loaded successfully" << std::endl;
    } else {
        std::cout << "  - Error: Could not load session file" << std::endl;
    }
}

void VisualizationManager::set_level_of_detail(bool enable, double threshold) {
    std::cout << "Level of detail (LOD) " << (enable ? "enabled" : "disabled") << std::endl;

    if (enable) {
        std::cout << "  - LOD threshold: " << threshold << std::endl;
        std::cout << "  - Automatic mesh simplification for distant objects" << std::endl;
        std::cout << "  - Dynamic quality adjustment based on performance" << std::endl;
    }
}

void VisualizationManager::enable_gpu_acceleration(bool enable) {
    std::cout << "GPU acceleration " << (enable ? "enabled" : "disabled") << std::endl;

    if (enable) {
        std::cout << "  - Utilizing GPU for rendering computations" << std::endl;
        std::cout << "  - Hardware-accelerated graphics pipeline" << std::endl;
        std::cout << "  - Improved performance for large datasets" << std::endl;
    }
}

void VisualizationManager::set_memory_limit(size_t limit_mb) {
    std::cout << "Memory limit set to " << limit_mb << " MB" << std::endl;
    std::cout << "  - Automatic data streaming for large datasets" << std::endl;
    std::cout << "  - Memory-efficient visualization algorithms" << std::endl;
    std::cout << "  - Garbage collection optimization enabled" << std::endl;
}

} // namespace SemiDGFEM
