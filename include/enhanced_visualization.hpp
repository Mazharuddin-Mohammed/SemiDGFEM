/**
 * Enhanced Visualization Module
 * 
 * This module provides advanced visualization capabilities for the SemiDGFEM simulator:
 * - 3D mesh and solution visualization
 * - Interactive plotting with VTK integration
 * - Real-time animation of transient solutions
 * - Advanced colormap and rendering options
 * - Multi-physics visualization
 * - Virtual reality (VR) support
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef ENHANCED_VISUALIZATION_HPP
#define ENHANCED_VISUALIZATION_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>

namespace SemiDGFEM {

// Forward declarations
class Mesh;
class Solution;

/**
 * Visualization data types
 */
enum class VisualizationType {
    MESH_WIREFRAME,
    MESH_SURFACE,
    SOLUTION_CONTOUR,
    SOLUTION_SURFACE,
    VECTOR_FIELD,
    STREAMLINES,
    ISOSURFACES,
    VOLUME_RENDERING,
    PARTICLE_TRACES
};

enum class ColorMapType {
    VIRIDIS,
    PLASMA,
    INFERNO,
    MAGMA,
    JET,
    RAINBOW,
    COOLWARM,
    SEISMIC,
    CUSTOM
};

enum class RenderingMode {
    WIREFRAME,
    SURFACE,
    POINTS,
    VOLUME,
    TRANSPARENT,
    SOLID
};

enum class CameraType {
    PERSPECTIVE,
    ORTHOGRAPHIC,
    FISHEYE,
    VR_STEREO
};

enum class LightingModel {
    PHONG,
    BLINN_PHONG,
    PHYSICALLY_BASED,
    AMBIENT_ONLY,
    CUSTOM
};

/**
 * Visualization configuration structures
 */
struct ColorMapConfig {
    ColorMapType type;
    double min_value;
    double max_value;
    bool logarithmic_scale;
    std::vector<double> custom_colors;
    double opacity;
    bool reverse_colormap;
    
    ColorMapConfig() : type(ColorMapType::VIRIDIS), min_value(0.0), max_value(1.0),
                      logarithmic_scale(false), opacity(1.0), reverse_colormap(false) {}
};

struct RenderingConfig {
    RenderingMode mode;
    LightingModel lighting;
    double ambient_intensity;
    double diffuse_intensity;
    double specular_intensity;
    double shininess;
    bool shadows_enabled;
    bool anti_aliasing;
    int samples_per_pixel;
    
    RenderingConfig() : mode(RenderingMode::SURFACE), lighting(LightingModel::PHONG),
                       ambient_intensity(0.3), diffuse_intensity(0.7), specular_intensity(0.5),
                       shininess(32.0), shadows_enabled(true), anti_aliasing(true), samples_per_pixel(4) {}
};

struct CameraConfig {
    CameraType type;
    std::vector<double> position;
    std::vector<double> target;
    std::vector<double> up_vector;
    double field_of_view;
    double near_plane;
    double far_plane;
    bool auto_zoom;
    
    CameraConfig() : type(CameraType::PERSPECTIVE), position({0, 0, 5}), target({0, 0, 0}),
                    up_vector({0, 1, 0}), field_of_view(45.0), near_plane(0.1), far_plane(100.0),
                    auto_zoom(true) {}
};

struct AnimationConfig {
    double frame_rate;
    double duration;
    bool loop_animation;
    bool save_frames;
    std::string output_directory;
    std::string frame_format;
    int frame_width;
    int frame_height;
    
    AnimationConfig() : frame_rate(30.0), duration(5.0), loop_animation(true),
                       save_frames(false), frame_format("png"), frame_width(1920), frame_height(1080) {}
};

/**
 * 3D Mesh Visualizer
 */
class MeshVisualizer3D {
public:
    MeshVisualizer3D();
    ~MeshVisualizer3D();
    
    // Mesh visualization
    void set_mesh(const Mesh& mesh);
    void set_rendering_config(const RenderingConfig& config);
    void set_camera_config(const CameraConfig& config);
    
    // Visualization methods
    void show_wireframe(bool show_edges = true, bool show_vertices = false);
    void show_surface(bool show_faces = true, bool show_normals = false);
    void show_element_quality(const std::vector<double>& quality_metrics);
    void show_boundary_conditions(const std::map<int, std::string>& bc_labels);
    
    // Material visualization
    void show_material_regions(const std::vector<int>& material_ids);
    void show_doping_profile(const std::vector<double>& doping_concentration);
    
    // Interactive features
    void enable_picking(bool enable = true);
    void enable_measurement_tools(bool enable = true);
    void enable_cross_sections(bool enable = true);
    
    // Export and rendering
    void render_to_file(const std::string& filename, int width = 1920, int height = 1080);
    void start_interactive_session();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Solution Visualizer
 */
class SolutionVisualizer {
public:
    SolutionVisualizer();
    ~SolutionVisualizer();
    
    // Solution data
    void set_solution(const Solution& solution);
    void set_mesh(const Mesh& mesh);
    void set_colormap_config(const ColorMapConfig& config);
    
    // Visualization methods
    void show_contour_plot(const std::string& field_name, int num_levels = 20);
    void show_surface_plot(const std::string& field_name);
    void show_vector_field(const std::string& vector_field_name, double scale = 1.0);
    void show_streamlines(const std::string& vector_field_name, int num_seeds = 100);
    void show_isosurfaces(const std::string& field_name, const std::vector<double>& iso_values);
    
    // Multi-field visualization
    void show_multi_field_comparison(const std::vector<std::string>& field_names);
    void show_field_difference(const std::string& field1, const std::string& field2);
    void show_field_ratio(const std::string& numerator, const std::string& denominator);
    
    // Advanced visualization
    void show_volume_rendering(const std::string& field_name, double opacity = 0.5);
    void show_particle_traces(const std::string& vector_field_name, int num_particles = 1000);
    void show_field_lines(const std::string& vector_field_name, const std::vector<double>& seed_points);
    
    // Cross-sections and slicing
    void add_cross_section(const std::vector<double>& plane_normal, double plane_distance);
    void add_slice_plane(const std::vector<double>& plane_point, const std::vector<double>& plane_normal);
    void show_line_plot(const std::vector<double>& start_point, const std::vector<double>& end_point);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Animation Engine
 */
class AnimationEngine {
public:
    AnimationEngine();
    ~AnimationEngine();
    
    // Animation setup
    void set_animation_config(const AnimationConfig& config);
    void add_time_series_data(const std::vector<Solution>& solutions, const std::vector<double>& time_points);
    
    // Animation types
    void create_solution_evolution_animation(const std::string& field_name);
    void create_mesh_refinement_animation(const std::vector<Mesh>& meshes);
    void create_parameter_sweep_animation(const std::vector<Solution>& solutions, 
                                        const std::vector<double>& parameter_values);
    
    // Camera animation
    void add_camera_keyframe(double time, const CameraConfig& camera);
    void create_orbit_animation(const std::vector<double>& center, double radius, double duration);
    void create_flythrough_animation(const std::vector<std::vector<double>>& waypoints);
    
    // Rendering and export
    void render_animation();
    void export_video(const std::string& filename, const std::string& codec = "h264");
    void export_gif(const std::string& filename, int quality = 80);
    
    // Real-time animation
    void start_real_time_animation();
    void pause_animation();
    void resume_animation();
    void stop_animation();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Interactive Plotter
 */
class InteractivePlotter {
public:
    InteractivePlotter();
    ~InteractivePlotter();
    
    // Plot types
    void create_line_plot(const std::vector<double>& x, const std::vector<double>& y, 
                         const std::string& label = "");
    void create_scatter_plot(const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& sizes = {});
    void create_surface_plot(const std::vector<std::vector<double>>& z_data);
    void create_heatmap(const std::vector<std::vector<double>>& data);
    void create_histogram(const std::vector<double>& data, int bins = 50);
    
    // Multi-plot layouts
    void create_subplot_grid(int rows, int cols);
    void set_active_subplot(int row, int col);
    void create_dashboard(const std::vector<std::string>& plot_types);
    
    // Customization
    void set_title(const std::string& title);
    void set_axis_labels(const std::string& x_label, const std::string& y_label, 
                        const std::string& z_label = "");
    void set_axis_limits(double x_min, double x_max, double y_min, double y_max);
    void set_colormap(const ColorMapConfig& config);
    void add_legend(bool show = true);
    void add_grid(bool show = true);
    
    // Interactive features
    void enable_zoom(bool enable = true);
    void enable_pan(bool enable = true);
    void enable_data_cursor(bool enable = true);
    void add_annotation(const std::string& text, double x, double y);
    
    // Export
    void save_plot(const std::string& filename, int dpi = 300);
    void show_plot();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * VR Visualizer
 */
class VRVisualizer {
public:
    VRVisualizer();
    ~VRVisualizer();
    
    // VR setup
    bool initialize_vr_system();
    void set_vr_rendering_config(const RenderingConfig& config);
    
    // VR visualization
    void load_mesh_in_vr(const Mesh& mesh);
    void load_solution_in_vr(const Solution& solution);
    void enable_hand_tracking(bool enable = true);
    void enable_gesture_recognition(bool enable = true);
    
    // VR interactions
    void add_vr_menu(const std::vector<std::string>& menu_items);
    void enable_teleportation(bool enable = true);
    void enable_object_manipulation(bool enable = true);
    
    // VR session
    void start_vr_session();
    void end_vr_session();
    bool is_vr_active() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Visualization Manager
 */
class VisualizationManager {
public:
    VisualizationManager();
    ~VisualizationManager();
    
    // Component access
    MeshVisualizer3D& get_mesh_visualizer();
    SolutionVisualizer& get_solution_visualizer();
    AnimationEngine& get_animation_engine();
    InteractivePlotter& get_plotter();
    VRVisualizer& get_vr_visualizer();
    
    // Unified interface
    void create_comprehensive_visualization(const Mesh& mesh, const Solution& solution);
    void create_multi_physics_dashboard(const std::vector<Solution>& solutions,
                                      const std::vector<std::string>& physics_names);
    
    // Session management
    void save_visualization_session(const std::string& filename);
    void load_visualization_session(const std::string& filename);
    
    // Performance optimization
    void set_level_of_detail(bool enable = true, double threshold = 0.01);
    void enable_gpu_acceleration(bool enable = true);
    void set_memory_limit(size_t limit_mb = 4096);
    
private:
    std::unique_ptr<MeshVisualizer3D> mesh_visualizer_;
    std::unique_ptr<SolutionVisualizer> solution_visualizer_;
    std::unique_ptr<AnimationEngine> animation_engine_;
    std::unique_ptr<InteractivePlotter> plotter_;
    std::unique_ptr<VRVisualizer> vr_visualizer_;
};

} // namespace SemiDGFEM

#endif // ENHANCED_VISUALIZATION_HPP
