/**
 * Machine Learning Integration for Semiconductor Device Simulation
 * 
 * This header defines machine learning capabilities including:
 * - Neural network-based device parameter prediction
 * - AI-enhanced mesh adaptation
 * - Machine learning-accelerated solvers
 * - Data-driven material property prediction
 * - Automated device optimization
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <random>

namespace SemiDGFEM {
namespace MachineLearning {

// Forward declarations
struct DeviceParameters;
struct MeshGeometry;
struct SimulationResults;

// ============================================================================
// Neural Network Framework
// ============================================================================

/**
 * @brief Activation functions for neural networks
 */
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU,
    ELU,
    SWISH,
    GELU
};

/**
 * @brief Optimization algorithms for training
 */
enum class OptimizerType {
    SGD,
    ADAM,
    RMSPROP,
    ADAGRAD,
    ADAMW
};

/**
 * @brief Loss functions for training
 */
enum class LossFunction {
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    HUBER_LOSS,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
};

/**
 * @brief Neural network layer configuration
 */
struct LayerConfig {
    size_t input_size;
    size_t output_size;
    ActivationFunction activation;
    double dropout_rate = 0.0;
    bool use_batch_norm = false;
    double weight_decay = 0.0;
};

/**
 * @brief Training configuration
 */
struct TrainingConfig {
    OptimizerType optimizer = OptimizerType::ADAM;
    LossFunction loss_function = LossFunction::MEAN_SQUARED_ERROR;
    double learning_rate = 0.001;
    double beta1 = 0.9;      // Adam parameter
    double beta2 = 0.999;    // Adam parameter
    double epsilon = 1e-8;   // Adam parameter
    size_t batch_size = 32;
    size_t max_epochs = 1000;
    double tolerance = 1e-6;
    double validation_split = 0.2;
    bool early_stopping = true;
    size_t patience = 50;
};

/**
 * @brief Neural network layer implementation
 */
class NeuralLayer {
private:
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    LayerConfig config_;
    
    // Batch normalization parameters
    std::vector<double> gamma_;
    std::vector<double> beta_;
    std::vector<double> running_mean_;
    std::vector<double> running_var_;
    
    // Optimizer state
    std::vector<std::vector<double>> weight_momentum_;
    std::vector<std::vector<double>> weight_velocity_;
    std::vector<double> bias_momentum_;
    std::vector<double> bias_velocity_;
    
public:
    NeuralLayer(const LayerConfig& config);
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& input, bool training = false);
    
    // Backward pass
    std::vector<double> backward(const std::vector<double>& gradient);
    
    // Parameter updates
    void update_parameters(const TrainingConfig& config, size_t iteration);
    
    // Getters
    const LayerConfig& get_config() const { return config_; }
    size_t get_parameter_count() const;
    
private:
    void initialize_weights();
    double apply_activation(double x, ActivationFunction func);
    double apply_activation_derivative(double x, ActivationFunction func);
    void apply_batch_normalization(std::vector<double>& values, bool training);
    void apply_dropout(std::vector<double>& values, bool training);
};

/**
 * @brief Multi-layer neural network
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<NeuralLayer>> layers_;
    TrainingConfig training_config_;
    std::mt19937 random_generator_;
    
    // Training history
    std::vector<double> training_loss_history_;
    std::vector<double> validation_loss_history_;
    
public:
    NeuralNetwork(const std::vector<LayerConfig>& layer_configs,
                  const TrainingConfig& training_config = TrainingConfig{});
    
    // Prediction
    std::vector<double> predict(const std::vector<double>& input);
    std::vector<std::vector<double>> predict_batch(const std::vector<std::vector<double>>& inputs);
    
    // Training
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y);
    
    double evaluate(const std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<double>>& y);
    
    // Model management
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Getters
    const std::vector<double>& get_training_history() const { return training_loss_history_; }
    const std::vector<double>& get_validation_history() const { return validation_loss_history_; }
    size_t get_total_parameters() const;
    
private:
    double compute_loss(const std::vector<double>& predicted, const std::vector<double>& actual);
    std::vector<double> compute_loss_gradient(const std::vector<double>& predicted, 
                                            const std::vector<double>& actual);
    void shuffle_data(std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y);
};

// ============================================================================
// Device Parameter Prediction
// ============================================================================

/**
 * @brief Device parameter prediction using neural networks
 */
struct DeviceFeatures {
    // Geometric features
    double channel_length;
    double channel_width;
    double oxide_thickness;
    double junction_depth;
    
    // Material features
    double substrate_doping;
    double source_drain_doping;
    double gate_work_function;
    double oxide_permittivity;
    
    // Operating conditions
    double temperature;
    double gate_voltage;
    double drain_voltage;
    double source_voltage;
    
    // Process variations
    double line_edge_roughness;
    double oxide_thickness_variation;
    double doping_fluctuation;
    double work_function_variation;
};

/**
 * @brief Predicted device parameters
 */
struct PredictedParameters {
    double threshold_voltage;
    double subthreshold_slope;
    double drain_induced_barrier_lowering;
    double transconductance;
    double output_conductance;
    double gate_leakage_current;
    double junction_leakage_current;
    double channel_mobility;
    double saturation_velocity;
    double short_channel_effect;
};

/**
 * @brief Device parameter predictor using neural networks
 */
class DeviceParameterPredictor {
private:
    std::unique_ptr<NeuralNetwork> network_;
    std::vector<double> feature_means_;
    std::vector<double> feature_stds_;
    std::vector<double> target_means_;
    std::vector<double> target_stds_;
    bool is_trained_;
    
public:
    DeviceParameterPredictor();
    
    // Training
    void train_predictor(const std::vector<DeviceFeatures>& features,
                        const std::vector<PredictedParameters>& parameters);
    
    // Prediction
    PredictedParameters predict_parameters(const DeviceFeatures& features);
    std::vector<PredictedParameters> predict_batch(const std::vector<DeviceFeatures>& features);
    
    // Model management
    void save_predictor(const std::string& filename) const;
    void load_predictor(const std::string& filename);
    
    // Validation
    double validate_predictor(const std::vector<DeviceFeatures>& test_features,
                             const std::vector<PredictedParameters>& test_parameters);
    
    bool is_trained() const { return is_trained_; }
    
private:
    std::vector<double> features_to_vector(const DeviceFeatures& features);
    std::vector<double> parameters_to_vector(const PredictedParameters& parameters);
    DeviceFeatures vector_to_features(const std::vector<double>& vec);
    PredictedParameters vector_to_parameters(const std::vector<double>& vec);
    void normalize_features(std::vector<std::vector<double>>& features);
    void normalize_targets(std::vector<std::vector<double>>& targets);
    std::vector<double> normalize_input(const std::vector<double>& input);
    std::vector<double> denormalize_output(const std::vector<double>& output);
};

// ============================================================================
// AI-Enhanced Mesh Adaptation
// ============================================================================

/**
 * @brief Mesh quality metrics for ML-based adaptation
 */
struct MeshQualityMetrics {
    double aspect_ratio_quality;
    double skewness_quality;
    double orthogonality_quality;
    double smoothness_quality;
    double solution_gradient_magnitude;
    double error_indicator;
    double refinement_efficiency;
    double computational_cost;
};

/**
 * @brief Mesh adaptation decision
 */
enum class AdaptationDecision {
    NO_CHANGE,
    REFINE,
    COARSEN,
    ANISOTROPIC_REFINE,
    RELOCATE
};

/**
 * @brief AI-enhanced mesh adaptation using reinforcement learning
 */
class AIEnhancedMeshAdapter {
private:
    std::unique_ptr<NeuralNetwork> policy_network_;
    std::unique_ptr<NeuralNetwork> value_network_;
    std::vector<MeshQualityMetrics> experience_buffer_;
    std::vector<AdaptationDecision> action_buffer_;
    std::vector<double> reward_buffer_;
    bool is_trained_;
    
public:
    AIEnhancedMeshAdapter();
    
    // Mesh adaptation
    AdaptationDecision suggest_adaptation(const MeshQualityMetrics& metrics);
    std::vector<AdaptationDecision> suggest_batch_adaptation(
        const std::vector<MeshQualityMetrics>& metrics);
    
    // Training
    void train_adapter(const std::vector<MeshQualityMetrics>& training_metrics,
                      const std::vector<AdaptationDecision>& optimal_decisions,
                      const std::vector<double>& rewards);
    
    // Experience collection
    void add_experience(const MeshQualityMetrics& metrics,
                       AdaptationDecision action,
                       double reward);
    
    void train_from_experience();
    
    // Model management
    void save_adapter(const std::string& filename) const;
    void load_adapter(const std::string& filename);
    
    bool is_trained() const { return is_trained_; }
    
private:
    std::vector<double> metrics_to_vector(const MeshQualityMetrics& metrics);
    std::vector<double> action_to_vector(AdaptationDecision action);
    AdaptationDecision vector_to_action(const std::vector<double>& vec);
    double compute_reward(const MeshQualityMetrics& old_metrics,
                         const MeshQualityMetrics& new_metrics,
                         AdaptationDecision action);
};

// ============================================================================
// ML-Accelerated Solvers
// ============================================================================

/**
 * @brief Solver acceleration using neural network preconditioners
 */
class MLAcceleratedSolver {
private:
    std::unique_ptr<NeuralNetwork> preconditioner_network_;
    std::unique_ptr<NeuralNetwork> convergence_predictor_;
    bool is_trained_;
    
public:
    MLAcceleratedSolver();
    
    // Preconditioning
    std::vector<double> apply_ml_preconditioner(const std::vector<double>& residual);
    
    // Convergence prediction
    double predict_convergence_rate(const std::vector<double>& residual_history);
    size_t predict_iterations_to_convergence(const std::vector<double>& residual_history);
    
    // Training
    void train_preconditioner(const std::vector<std::vector<double>>& residuals,
                             const std::vector<std::vector<double>>& preconditioned);
    
    void train_convergence_predictor(const std::vector<std::vector<double>>& residual_histories,
                                   const std::vector<size_t>& iteration_counts);
    
    // Model management
    void save_solver(const std::string& filename) const;
    void load_solver(const std::string& filename);
    
    bool is_trained() const { return is_trained_; }
};

} // namespace MachineLearning
} // namespace SemiDGFEM
