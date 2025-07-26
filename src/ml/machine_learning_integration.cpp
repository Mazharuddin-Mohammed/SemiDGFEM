/**
 * Machine Learning Integration for Semiconductor Device Simulation
 * 
 * This module implements machine learning capabilities including:
 * - Neural network-based device parameter prediction
 * - AI-enhanced mesh adaptation
 * - Machine learning-accelerated solvers
 * - Data-driven material property prediction
 * - Automated device optimization
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#include "machine_learning_integration.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <stdexcept>

namespace SemiDGFEM {
namespace MachineLearning {

// ============================================================================
// Neural Layer Implementation
// ============================================================================

NeuralLayer::NeuralLayer(const LayerConfig& config) : config_(config) {
    initialize_weights();
    
    // Initialize batch normalization parameters
    if (config_.use_batch_norm) {
        gamma_.resize(config_.output_size, 1.0);
        beta_.resize(config_.output_size, 0.0);
        running_mean_.resize(config_.output_size, 0.0);
        running_var_.resize(config_.output_size, 1.0);
    }
    
    // Initialize optimizer state
    weight_momentum_.resize(config_.output_size, std::vector<double>(config_.input_size, 0.0));
    weight_velocity_.resize(config_.output_size, std::vector<double>(config_.input_size, 0.0));
    bias_momentum_.resize(config_.output_size, 0.0);
    bias_velocity_.resize(config_.output_size, 0.0);
}

void NeuralLayer::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization
    double limit = std::sqrt(6.0 / (config_.input_size + config_.output_size));
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    weights_.resize(config_.output_size);
    for (size_t i = 0; i < config_.output_size; ++i) {
        weights_[i].resize(config_.input_size);
        for (size_t j = 0; j < config_.input_size; ++j) {
            weights_[i][j] = dist(gen);
        }
    }
    
    // Initialize biases to zero
    biases_.resize(config_.output_size, 0.0);
}

std::vector<double> NeuralLayer::forward(const std::vector<double>& input, bool training) {
    if (input.size() != config_.input_size) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    std::vector<double> output(config_.output_size);
    
    // Linear transformation: output = weights * input + bias
    for (size_t i = 0; i < config_.output_size; ++i) {
        output[i] = biases_[i];
        for (size_t j = 0; j < config_.input_size; ++j) {
            output[i] += weights_[i][j] * input[j];
        }
    }
    
    // Apply batch normalization
    if (config_.use_batch_norm) {
        apply_batch_normalization(output, training);
    }
    
    // Apply activation function
    for (size_t i = 0; i < config_.output_size; ++i) {
        output[i] = apply_activation(output[i], config_.activation);
    }
    
    // Apply dropout during training
    if (training && config_.dropout_rate > 0.0) {
        apply_dropout(output, training);
    }
    
    return output;
}

double NeuralLayer::apply_activation(double x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return std::max(0.0, x);
        case ActivationFunction::SIGMOID:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationFunction::TANH:
            return std::tanh(x);
        case ActivationFunction::LEAKY_RELU:
            return x > 0 ? x : 0.01 * x;
        case ActivationFunction::ELU:
            return x > 0 ? x : std::exp(x) - 1.0;
        case ActivationFunction::SWISH:
            return x / (1.0 + std::exp(-x));
        case ActivationFunction::GELU:
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        default:
            return x;
    }
}

double NeuralLayer::apply_activation_derivative(double x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return x > 0 ? 1.0 : 0.0;
        case ActivationFunction::SIGMOID: {
            double s = apply_activation(x, func);
            return s * (1.0 - s);
        }
        case ActivationFunction::TANH: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        case ActivationFunction::LEAKY_RELU:
            return x > 0 ? 1.0 : 0.01;
        case ActivationFunction::ELU:
            return x > 0 ? 1.0 : std::exp(x);
        case ActivationFunction::SWISH: {
            double sigmoid = 1.0 / (1.0 + std::exp(-x));
            return sigmoid + x * sigmoid * (1.0 - sigmoid);
        }
        case ActivationFunction::GELU: {
            double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
            double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
            return cdf + x * pdf;
        }
        default:
            return 1.0;
    }
}

void NeuralLayer::apply_batch_normalization(std::vector<double>& values, bool training) {
    if (!config_.use_batch_norm) return;
    
    if (training) {
        // Compute batch statistics
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        
        // Update running statistics
        double momentum = 0.9;
        for (size_t i = 0; i < running_mean_.size(); ++i) {
            running_mean_[i] = momentum * running_mean_[i] + (1.0 - momentum) * mean;
            running_var_[i] = momentum * running_var_[i] + (1.0 - momentum) * variance;
        }
        
        // Normalize using batch statistics
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = gamma_[i] * (values[i] - mean) / std::sqrt(variance + 1e-8) + beta_[i];
        }
    } else {
        // Normalize using running statistics
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = gamma_[i] * (values[i] - running_mean_[i]) / 
                       std::sqrt(running_var_[i] + 1e-8) + beta_[i];
        }
    }
}

void NeuralLayer::apply_dropout(std::vector<double>& values, bool training) {
    if (!training || config_.dropout_rate <= 0.0) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    double scale = 1.0 / (1.0 - config_.dropout_rate);
    for (double& val : values) {
        if (dist(gen) < config_.dropout_rate) {
            val = 0.0;
        } else {
            val *= scale;
        }
    }
}

size_t NeuralLayer::get_parameter_count() const {
    size_t count = config_.input_size * config_.output_size + config_.output_size; // weights + biases
    if (config_.use_batch_norm) {
        count += 2 * config_.output_size; // gamma + beta
    }
    return count;
}

// ============================================================================
// Neural Network Implementation
// ============================================================================

NeuralNetwork::NeuralNetwork(const std::vector<LayerConfig>& layer_configs,
                           const TrainingConfig& training_config)
    : training_config_(training_config), random_generator_(std::random_device{}()) {
    
    for (const auto& config : layer_configs) {
        layers_.push_back(std::make_unique<NeuralLayer>(config));
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> current_input = input;
    
    for (const auto& layer : layers_) {
        current_input = layer->forward(current_input, false); // inference mode
    }
    
    return current_input;
}

std::vector<std::vector<double>> NeuralNetwork::predict_batch(
    const std::vector<std::vector<double>>& inputs) {
    
    std::vector<std::vector<double>> predictions;
    predictions.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        predictions.push_back(predict(input));
    }
    
    return predictions;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X,
                         const std::vector<std::vector<double>>& y) {
    
    if (X.size() != y.size()) {
        throw std::invalid_argument("Input and target sizes must match");
    }
    
    // Split data into training and validation sets
    size_t total_samples = X.size();
    size_t validation_size = static_cast<size_t>(total_samples * training_config_.validation_split);
    size_t training_size = total_samples - validation_size;
    
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + training_size);
    std::vector<std::vector<double>> y_train(y.begin(), y.begin() + training_size);
    std::vector<std::vector<double>> X_val(X.begin() + training_size, X.end());
    std::vector<std::vector<double>> y_val(y.begin() + training_size, y.end());
    
    double best_val_loss = std::numeric_limits<double>::max();
    size_t patience_counter = 0;
    
    for (size_t epoch = 0; epoch < training_config_.max_epochs; ++epoch) {
        // Shuffle training data
        auto X_train_copy = X_train;
        auto y_train_copy = y_train;
        shuffle_data(X_train_copy, y_train_copy);
        
        // Training phase
        double epoch_loss = 0.0;
        size_t num_batches = (training_size + training_config_.batch_size - 1) / training_config_.batch_size;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start_idx = batch * training_config_.batch_size;
            size_t end_idx = std::min(start_idx + training_config_.batch_size, training_size);
            
            double batch_loss = 0.0;
            for (size_t i = start_idx; i < end_idx; ++i) {
                auto prediction = predict(X_train_copy[i]);
                batch_loss += compute_loss(prediction, y_train_copy[i]);
            }
            
            epoch_loss += batch_loss / (end_idx - start_idx);
        }
        
        epoch_loss /= num_batches;
        training_loss_history_.push_back(epoch_loss);
        
        // Validation phase
        double val_loss = evaluate(X_val, y_val);
        validation_loss_history_.push_back(val_loss);
        
        // Early stopping
        if (training_config_.early_stopping) {
            if (val_loss < best_val_loss - training_config_.tolerance) {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= training_config_.patience) {
                    break;
                }
            }
        }
        
        // Check convergence
        if (epoch_loss < training_config_.tolerance) {
            break;
        }
    }
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& X,
                              const std::vector<std::vector<double>>& y) {
    
    double total_loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto prediction = predict(X[i]);
        total_loss += compute_loss(prediction, y[i]);
    }
    
    return total_loss / X.size();
}

double NeuralNetwork::compute_loss(const std::vector<double>& predicted, 
                                  const std::vector<double>& actual) {
    
    if (predicted.size() != actual.size()) {
        throw std::invalid_argument("Prediction and actual sizes must match");
    }
    
    double loss = 0.0;
    
    switch (training_config_.loss_function) {
        case LossFunction::MEAN_SQUARED_ERROR: {
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = predicted[i] - actual[i];
                loss += diff * diff;
            }
            loss /= predicted.size();
            break;
        }
        case LossFunction::MEAN_ABSOLUTE_ERROR: {
            for (size_t i = 0; i < predicted.size(); ++i) {
                loss += std::abs(predicted[i] - actual[i]);
            }
            loss /= predicted.size();
            break;
        }
        case LossFunction::HUBER_LOSS: {
            double delta = 1.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = std::abs(predicted[i] - actual[i]);
                if (diff <= delta) {
                    loss += 0.5 * diff * diff;
                } else {
                    loss += delta * (diff - 0.5 * delta);
                }
            }
            loss /= predicted.size();
            break;
        }
        default:
            // Default to MSE
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = predicted[i] - actual[i];
                loss += diff * diff;
            }
            loss /= predicted.size();
            break;
    }
    
    return loss;
}

void NeuralNetwork::shuffle_data(std::vector<std::vector<double>>& X, 
                                std::vector<std::vector<double>>& y) {
    
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), random_generator_);
    
    auto X_copy = X;
    auto y_copy = y;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        X[i] = X_copy[indices[i]];
        y[i] = y_copy[indices[i]];
    }
}

size_t NeuralNetwork::get_total_parameters() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        total += layer->get_parameter_count();
    }
    return total;
}

// ============================================================================
// Device Parameter Predictor Implementation
// ============================================================================

DeviceParameterPredictor::DeviceParameterPredictor() : is_trained_(false) {
    // Create neural network architecture for device parameter prediction
    std::vector<LayerConfig> layer_configs = {
        {16, 64, ActivationFunction::RELU, 0.1, true},    // Input layer
        {64, 128, ActivationFunction::RELU, 0.2, true},   // Hidden layer 1
        {128, 64, ActivationFunction::RELU, 0.2, true},   // Hidden layer 2
        {64, 32, ActivationFunction::RELU, 0.1, true},    // Hidden layer 3
        {32, 10, ActivationFunction::RELU, 0.0, false}    // Output layer
    };

    TrainingConfig training_config;
    training_config.learning_rate = 0.001;
    training_config.batch_size = 32;
    training_config.max_epochs = 1000;
    training_config.early_stopping = true;
    training_config.patience = 50;

    network_ = std::make_unique<NeuralNetwork>(layer_configs, training_config);
}

void DeviceParameterPredictor::train_predictor(
    const std::vector<DeviceFeatures>& features,
    const std::vector<PredictedParameters>& parameters) {

    if (features.size() != parameters.size()) {
        throw std::invalid_argument("Features and parameters size mismatch");
    }

    // Convert to vectors
    std::vector<std::vector<double>> X, y;
    for (size_t i = 0; i < features.size(); ++i) {
        X.push_back(features_to_vector(features[i]));
        y.push_back(parameters_to_vector(parameters[i]));
    }

    // Normalize data
    normalize_features(X);
    normalize_targets(y);

    // Train the network
    network_->train(X, y);
    is_trained_ = true;
}

PredictedParameters DeviceParameterPredictor::predict_parameters(const DeviceFeatures& features) {
    if (!is_trained_) {
        throw std::runtime_error("Predictor must be trained before prediction");
    }

    auto input = normalize_input(features_to_vector(features));
    auto output = network_->predict(input);
    auto denormalized = denormalize_output(output);

    return vector_to_parameters(denormalized);
}

std::vector<double> DeviceParameterPredictor::features_to_vector(const DeviceFeatures& features) {
    return {
        features.channel_length,
        features.channel_width,
        features.oxide_thickness,
        features.junction_depth,
        features.substrate_doping,
        features.source_drain_doping,
        features.gate_work_function,
        features.oxide_permittivity,
        features.temperature,
        features.gate_voltage,
        features.drain_voltage,
        features.source_voltage,
        features.line_edge_roughness,
        features.oxide_thickness_variation,
        features.doping_fluctuation,
        features.work_function_variation
    };
}

std::vector<double> DeviceParameterPredictor::parameters_to_vector(const PredictedParameters& parameters) {
    return {
        parameters.threshold_voltage,
        parameters.subthreshold_slope,
        parameters.drain_induced_barrier_lowering,
        parameters.transconductance,
        parameters.output_conductance,
        parameters.gate_leakage_current,
        parameters.junction_leakage_current,
        parameters.channel_mobility,
        parameters.saturation_velocity,
        parameters.short_channel_effect
    };
}

PredictedParameters DeviceParameterPredictor::vector_to_parameters(const std::vector<double>& vec) {
    if (vec.size() != 10) {
        throw std::invalid_argument("Parameter vector must have 10 elements");
    }

    PredictedParameters params;
    params.threshold_voltage = vec[0];
    params.subthreshold_slope = vec[1];
    params.drain_induced_barrier_lowering = vec[2];
    params.transconductance = vec[3];
    params.output_conductance = vec[4];
    params.gate_leakage_current = vec[5];
    params.junction_leakage_current = vec[6];
    params.channel_mobility = vec[7];
    params.saturation_velocity = vec[8];
    params.short_channel_effect = vec[9];

    return params;
}

void DeviceParameterPredictor::normalize_features(std::vector<std::vector<double>>& features) {
    if (features.empty()) return;

    size_t num_features = features[0].size();
    feature_means_.resize(num_features, 0.0);
    feature_stds_.resize(num_features, 0.0);

    // Compute means
    for (const auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            feature_means_[i] += sample[i];
        }
    }
    for (double& mean : feature_means_) {
        mean /= features.size();
    }

    // Compute standard deviations
    for (const auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            double diff = sample[i] - feature_means_[i];
            feature_stds_[i] += diff * diff;
        }
    }
    for (double& std : feature_stds_) {
        std = std::sqrt(std / features.size());
        if (std < 1e-8) std = 1.0; // Avoid division by zero
    }

    // Normalize features
    for (auto& sample : features) {
        for (size_t i = 0; i < num_features; ++i) {
            sample[i] = (sample[i] - feature_means_[i]) / feature_stds_[i];
        }
    }
}

void DeviceParameterPredictor::normalize_targets(std::vector<std::vector<double>>& targets) {
    if (targets.empty()) return;

    size_t num_targets = targets[0].size();
    target_means_.resize(num_targets, 0.0);
    target_stds_.resize(num_targets, 0.0);

    // Compute means
    for (const auto& sample : targets) {
        for (size_t i = 0; i < num_targets; ++i) {
            target_means_[i] += sample[i];
        }
    }
    for (double& mean : target_means_) {
        mean /= targets.size();
    }

    // Compute standard deviations
    for (const auto& sample : targets) {
        for (size_t i = 0; i < num_targets; ++i) {
            double diff = sample[i] - target_means_[i];
            target_stds_[i] += diff * diff;
        }
    }
    for (double& std : target_stds_) {
        std = std::sqrt(std / targets.size());
        if (std < 1e-8) std = 1.0; // Avoid division by zero
    }

    // Normalize targets
    for (auto& sample : targets) {
        for (size_t i = 0; i < num_targets; ++i) {
            sample[i] = (sample[i] - target_means_[i]) / target_stds_[i];
        }
    }
}

std::vector<double> DeviceParameterPredictor::normalize_input(const std::vector<double>& input) {
    std::vector<double> normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = (input[i] - feature_means_[i]) / feature_stds_[i];
    }
    return normalized;
}

std::vector<double> DeviceParameterPredictor::denormalize_output(const std::vector<double>& output) {
    std::vector<double> denormalized(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        denormalized[i] = output[i] * target_stds_[i] + target_means_[i];
    }
    return denormalized;
}

} // namespace MachineLearning
} // namespace SemiDGFEM
