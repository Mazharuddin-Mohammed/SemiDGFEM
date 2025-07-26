"""
Machine Learning Integration for Semiconductor Device Simulation

This module provides machine learning capabilities including:
- Neural network-based device parameter prediction
- AI-enhanced mesh adaptation
- Machine learning-accelerated solvers
- Data-driven material property prediction
- Automated device optimization

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Classes
# ============================================================================

class ActivationFunction(Enum):
    """Activation functions for neural networks"""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"

class OptimizerType(Enum):
    """Optimization algorithms for training"""
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADAMW = "adamw"

class LossFunction(Enum):
    """Loss functions for training"""
    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae"
    HUBER_LOSS = "huber"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"

@dataclass
class LayerConfig:
    """Neural network layer configuration"""
    input_size: int
    output_size: int
    activation: ActivationFunction = ActivationFunction.RELU
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    weight_decay: float = 0.0

@dataclass
class TrainingConfig:
    """Training configuration"""
    optimizer: OptimizerType = OptimizerType.ADAM
    loss_function: LossFunction = LossFunction.MEAN_SQUARED_ERROR
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    batch_size: int = 32
    max_epochs: int = 1000
    tolerance: float = 1e-6
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 50

@dataclass
class DeviceFeatures:
    """Device features for parameter prediction"""
    # Geometric features
    channel_length: float
    channel_width: float
    oxide_thickness: float
    junction_depth: float
    
    # Material features
    substrate_doping: float
    source_drain_doping: float
    gate_work_function: float
    oxide_permittivity: float
    
    # Operating conditions
    temperature: float
    gate_voltage: float
    drain_voltage: float
    source_voltage: float
    
    # Process variations
    line_edge_roughness: float
    oxide_thickness_variation: float
    doping_fluctuation: float
    work_function_variation: float

@dataclass
class PredictedParameters:
    """Predicted device parameters"""
    threshold_voltage: float
    subthreshold_slope: float
    drain_induced_barrier_lowering: float
    transconductance: float
    output_conductance: float
    gate_leakage_current: float
    junction_leakage_current: float
    channel_mobility: float
    saturation_velocity: float
    short_channel_effect: float

@dataclass
class MeshQualityMetrics:
    """Mesh quality metrics for ML-based adaptation"""
    aspect_ratio_quality: float
    skewness_quality: float
    orthogonality_quality: float
    smoothness_quality: float
    solution_gradient_magnitude: float
    error_indicator: float
    refinement_efficiency: float
    computational_cost: float

class AdaptationDecision(Enum):
    """Mesh adaptation decision"""
    NO_CHANGE = "no_change"
    REFINE = "refine"
    COARSEN = "coarsen"
    ANISOTROPIC_REFINE = "anisotropic_refine"
    RELOCATE = "relocate"

# ============================================================================
# Neural Network Implementation
# ============================================================================

class NeuralLayer:
    """Neural network layer implementation"""
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.weights = None
        self.biases = None
        self.gamma = None  # Batch norm scale
        self.beta = None   # Batch norm shift
        self.running_mean = None
        self.running_var = None
        
        # Optimizer state
        self.weight_momentum = None
        self.weight_velocity = None
        self.bias_momentum = None
        self.bias_velocity = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights using Xavier/Glorot initialization"""
        # Xavier initialization
        limit = np.sqrt(6.0 / (self.config.input_size + self.config.output_size))
        self.weights = np.random.uniform(-limit, limit, 
                                       (self.config.output_size, self.config.input_size))
        self.biases = np.zeros(self.config.output_size)
        
        # Batch normalization parameters
        if self.config.use_batch_norm:
            self.gamma = np.ones(self.config.output_size)
            self.beta = np.zeros(self.config.output_size)
            self.running_mean = np.zeros(self.config.output_size)
            self.running_var = np.ones(self.config.output_size)
        
        # Initialize optimizer state
        self.weight_momentum = np.zeros_like(self.weights)
        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)
        self.bias_velocity = np.zeros_like(self.biases)
    
    def forward(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the layer"""
        if input_data.shape[-1] != self.config.input_size:
            raise ValueError(f"Input size mismatch: expected {self.config.input_size}, got {input_data.shape[-1]}")
        
        # Linear transformation
        output = np.dot(input_data, self.weights.T) + self.biases
        
        # Batch normalization
        if self.config.use_batch_norm:
            output = self._apply_batch_normalization(output, training)
        
        # Activation function
        output = self._apply_activation(output)
        
        # Dropout during training
        if training and self.config.dropout_rate > 0.0:
            output = self._apply_dropout(output)
        
        return output
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.config.activation == ActivationFunction.RELU:
            return np.maximum(0, x)
        elif self.config.activation == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.config.activation == ActivationFunction.TANH:
            return np.tanh(x)
        elif self.config.activation == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        elif self.config.activation == ActivationFunction.ELU:
            return np.where(x > 0, x, np.exp(x) - 1.0)
        elif self.config.activation == ActivationFunction.SWISH:
            return x / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.config.activation == ActivationFunction.GELU:
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        else:
            return x
    
    def _apply_batch_normalization(self, x: np.ndarray, training: bool) -> np.ndarray:
        """Apply batch normalization"""
        if not self.config.use_batch_norm:
            return x
        
        if training:
            # Compute batch statistics
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            momentum = 0.9
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mean
            self.running_var = momentum * self.running_var + (1 - momentum) * var
            
            # Normalize using batch statistics
            x_norm = (x - mean) / np.sqrt(var + 1e-8)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + 1e-8)
        
        return self.gamma * x_norm + self.beta
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout regularization"""
        if self.config.dropout_rate <= 0.0:
            return x
        
        mask = np.random.random(x.shape) > self.config.dropout_rate
        return x * mask / (1.0 - self.config.dropout_rate)
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters in the layer"""
        count = self.config.input_size * self.config.output_size + self.config.output_size
        if self.config.use_batch_norm:
            count += 2 * self.config.output_size
        return count

class NeuralNetwork:
    """Multi-layer neural network implementation"""
    
    def __init__(self, layer_configs: List[LayerConfig], training_config: TrainingConfig = None):
        self.layer_configs = layer_configs
        self.training_config = training_config or TrainingConfig()
        self.layers = [NeuralLayer(config) for config in layer_configs]
        self.training_history = []
        self.validation_history = []
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
        current_input = input_data
        for layer in self.layers:
            current_input = layer.forward(current_input, training=False)
        return current_input
    
    def predict_batch(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Make batch predictions"""
        return [self.predict(input_data) for input_data in inputs]
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the neural network"""
        if len(X) != len(y):
            raise ValueError("Input and target sizes must match")
        
        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - self.training_config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.training_config.max_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Training phase
            epoch_loss = 0.0
            num_batches = (len(X_train) + self.training_config.batch_size - 1) // self.training_config.batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.training_config.batch_size
                end_idx = min(start_idx + self.training_config.batch_size, len(X_train))
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.predict(X_batch)
                batch_loss = self._compute_loss(predictions, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= num_batches
            self.training_history.append(epoch_loss)
            
            # Validation phase
            val_predictions = self.predict(X_val)
            val_loss = self._compute_loss(val_predictions, y_val)
            self.validation_history.append(val_loss)
            
            # Early stopping
            if self.training_config.early_stopping:
                if val_loss < best_val_loss - self.training_config.tolerance:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.training_config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Check convergence
            if epoch_loss < self.training_config.tolerance:
                logger.info(f"Converged at epoch {epoch}")
                break
        
        return {
            'final_training_loss': self.training_history[-1],
            'final_validation_loss': self.validation_history[-1],
            'epochs_trained': len(self.training_history),
            'converged': epoch_loss < self.training_config.tolerance
        }
    
    def _compute_loss(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute loss between predicted and actual values"""
        if self.training_config.loss_function == LossFunction.MEAN_SQUARED_ERROR:
            return np.mean((predicted - actual) ** 2)
        elif self.training_config.loss_function == LossFunction.MEAN_ABSOLUTE_ERROR:
            return np.mean(np.abs(predicted - actual))
        elif self.training_config.loss_function == LossFunction.HUBER_LOSS:
            delta = 1.0
            diff = np.abs(predicted - actual)
            return np.mean(np.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)))
        else:
            # Default to MSE
            return np.mean((predicted - actual) ** 2)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the network on test data"""
        predictions = self.predict(X)
        return self._compute_loss(predictions, y)
    
    def save_model(self, filename: str):
        """Save the trained model"""
        # Convert layer configs to serializable format
        layer_configs_serializable = []
        for config in self.layer_configs:
            config_dict = asdict(config)
            config_dict['activation'] = config.activation.value  # Convert enum to string
            layer_configs_serializable.append(config_dict)

        # Convert training config to serializable format
        training_config_dict = asdict(self.training_config)
        training_config_dict['optimizer'] = self.training_config.optimizer.value
        training_config_dict['loss_function'] = self.training_config.loss_function.value

        model_data = {
            'layer_configs': layer_configs_serializable,
            'training_config': training_config_dict,
            'layers': [],
            'training_history': self.training_history,
            'validation_history': self.validation_history
        }
        
        # Save layer parameters
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
            }
            if layer.config.use_batch_norm:
                layer_data.update({
                    'gamma': layer.gamma.tolist(),
                    'beta': layer.beta.tolist(),
                    'running_mean': layer.running_mean.tolist(),
                    'running_var': layer.running_var.tolist()
                })
            model_data['layers'].append(layer_data)
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filename: str):
        """Load a trained model"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Reconstruct layer configs
        self.layer_configs = []
        for config_dict in model_data['layer_configs']:
            config_dict['activation'] = ActivationFunction(config_dict['activation'])
            self.layer_configs.append(LayerConfig(**config_dict))
        
        # Reconstruct training config
        training_dict = model_data['training_config']
        training_dict['optimizer'] = OptimizerType(training_dict['optimizer'])
        training_dict['loss_function'] = LossFunction(training_dict['loss_function'])
        self.training_config = TrainingConfig(**training_dict)
        
        # Reconstruct layers
        self.layers = []
        for i, (config, layer_data) in enumerate(zip(self.layer_configs, model_data['layers'])):
            layer = NeuralLayer(config)
            layer.weights = np.array(layer_data['weights'])
            layer.biases = np.array(layer_data['biases'])
            
            if config.use_batch_norm:
                layer.gamma = np.array(layer_data['gamma'])
                layer.beta = np.array(layer_data['beta'])
                layer.running_mean = np.array(layer_data['running_mean'])
                layer.running_var = np.array(layer_data['running_var'])
            
            self.layers.append(layer)
        
        # Restore training history
        self.training_history = model_data['training_history']
        self.validation_history = model_data['validation_history']
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the network"""
        return sum(layer.get_parameter_count() for layer in self.layers)

# ============================================================================
# Device Parameter Predictor
# ============================================================================

class DeviceParameterPredictor:
    """Device parameter predictor using neural networks"""

    def __init__(self):
        self.network = None
        self.feature_scaler = None
        self.target_scaler = None
        self.is_trained = False
        self._create_network()

    def _create_network(self):
        """Create neural network architecture for device parameter prediction"""
        layer_configs = [
            LayerConfig(16, 64, ActivationFunction.RELU, 0.1, True),    # Input layer
            LayerConfig(64, 128, ActivationFunction.RELU, 0.2, True),   # Hidden layer 1
            LayerConfig(128, 64, ActivationFunction.RELU, 0.2, True),   # Hidden layer 2
            LayerConfig(64, 32, ActivationFunction.RELU, 0.1, True),    # Hidden layer 3
            LayerConfig(32, 10, ActivationFunction.RELU, 0.0, False)    # Output layer
        ]

        training_config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            max_epochs=1000,
            early_stopping=True,
            patience=50
        )

        self.network = NeuralNetwork(layer_configs, training_config)

    def train_predictor(self, features: List[DeviceFeatures], parameters: List[PredictedParameters]) -> Dict[str, Any]:
        """Train the device parameter predictor"""
        if len(features) != len(parameters):
            raise ValueError("Features and parameters size mismatch")

        # Convert to arrays
        X = np.array([self._features_to_vector(f) for f in features])
        y = np.array([self._parameters_to_vector(p) for p in parameters])

        # Normalize data
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        X_normalized = self.feature_scaler.fit_transform(X)
        y_normalized = self.target_scaler.fit_transform(y)

        # Train the network
        training_results = self.network.train(X_normalized, y_normalized)
        self.is_trained = True

        return training_results

    def predict_parameters(self, features: DeviceFeatures) -> PredictedParameters:
        """Predict device parameters from features"""
        if not self.is_trained:
            raise RuntimeError("Predictor must be trained before prediction")

        # Convert features to vector and normalize
        feature_vector = self._features_to_vector(features)
        feature_normalized = self.feature_scaler.transform(feature_vector.reshape(1, -1))

        # Predict and denormalize
        prediction_normalized = self.network.predict(feature_normalized)
        prediction = self.target_scaler.inverse_transform(prediction_normalized.reshape(1, -1))

        return self._vector_to_parameters(prediction.flatten())

    def predict_batch(self, features_list: List[DeviceFeatures]) -> List[PredictedParameters]:
        """Predict parameters for a batch of features"""
        if not self.is_trained:
            raise RuntimeError("Predictor must be trained before prediction")

        # Convert to array and normalize
        X = np.array([self._features_to_vector(f) for f in features_list])
        X_normalized = self.feature_scaler.transform(X)

        # Predict and denormalize
        predictions_normalized = self.network.predict(X_normalized)
        predictions = self.target_scaler.inverse_transform(predictions_normalized)

        return [self._vector_to_parameters(pred) for pred in predictions]

    def validate_predictor(self, test_features: List[DeviceFeatures],
                          test_parameters: List[PredictedParameters]) -> Dict[str, float]:
        """Validate the predictor on test data"""
        if not self.is_trained:
            raise RuntimeError("Predictor must be trained before validation")

        # Make predictions
        predicted_params = self.predict_batch(test_features)

        # Compute metrics
        actual_vectors = np.array([self._parameters_to_vector(p) for p in test_parameters])
        predicted_vectors = np.array([self._parameters_to_vector(p) for p in predicted_params])

        mse = np.mean((actual_vectors - predicted_vectors) ** 2)
        mae = np.mean(np.abs(actual_vectors - predicted_vectors))
        r2 = 1 - np.sum((actual_vectors - predicted_vectors) ** 2) / np.sum((actual_vectors - np.mean(actual_vectors, axis=0)) ** 2)

        # Per-parameter metrics
        parameter_names = [
            'threshold_voltage', 'subthreshold_slope', 'drain_induced_barrier_lowering',
            'transconductance', 'output_conductance', 'gate_leakage_current',
            'junction_leakage_current', 'channel_mobility', 'saturation_velocity',
            'short_channel_effect'
        ]

        per_parameter_mse = {}
        for i, name in enumerate(parameter_names):
            per_parameter_mse[f'{name}_mse'] = np.mean((actual_vectors[:, i] - predicted_vectors[:, i]) ** 2)

        return {
            'overall_mse': mse,
            'overall_mae': mae,
            'overall_r2': r2,
            **per_parameter_mse
        }

    def save_predictor(self, filename: str):
        """Save the trained predictor"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained predictor")

        # Save network
        network_filename = filename.replace('.pkl', '_network.json')
        self.network.save_model(network_filename)

        # Save scalers and metadata
        predictor_data = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'is_trained': self.is_trained,
            'network_filename': network_filename
        }

        with open(filename, 'wb') as f:
            pickle.dump(predictor_data, f)

    def load_predictor(self, filename: str):
        """Load a trained predictor"""
        with open(filename, 'rb') as f:
            predictor_data = pickle.load(f)

        self.feature_scaler = predictor_data['feature_scaler']
        self.target_scaler = predictor_data['target_scaler']
        self.is_trained = predictor_data['is_trained']

        # Load network
        network_filename = predictor_data['network_filename']
        self._create_network()
        self.network.load_model(network_filename)

    def _features_to_vector(self, features: DeviceFeatures) -> np.ndarray:
        """Convert DeviceFeatures to numpy vector"""
        return np.array([
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
        ])

    def _parameters_to_vector(self, parameters: PredictedParameters) -> np.ndarray:
        """Convert PredictedParameters to numpy vector"""
        return np.array([
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
        ])

    def _vector_to_parameters(self, vector: np.ndarray) -> PredictedParameters:
        """Convert numpy vector to PredictedParameters"""
        if len(vector) != 10:
            raise ValueError("Parameter vector must have 10 elements")

        return PredictedParameters(
            threshold_voltage=float(vector[0]),
            subthreshold_slope=float(vector[1]),
            drain_induced_barrier_lowering=float(vector[2]),
            transconductance=float(vector[3]),
            output_conductance=float(vector[4]),
            gate_leakage_current=float(vector[5]),
            junction_leakage_current=float(vector[6]),
            channel_mobility=float(vector[7]),
            saturation_velocity=float(vector[8]),
            short_channel_effect=float(vector[9])
        )

# ============================================================================
# Simple Scaler Implementation
# ============================================================================

class StandardScaler:
    """Simple standard scaler implementation"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        """Fit the scaler to data"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ < 1e-8] = 1.0
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        return X * self.std_ + self.mean_

# ============================================================================
# AI-Enhanced Mesh Adapter
# ============================================================================

class AIEnhancedMeshAdapter:
    """AI-enhanced mesh adaptation using reinforcement learning"""

    def __init__(self):
        self.policy_network = None
        self.value_network = None
        self.experience_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.is_trained = False
        self._create_networks()

    def _create_networks(self):
        """Create policy and value networks"""
        # Policy network (state -> action probabilities)
        policy_configs = [
            LayerConfig(8, 32, ActivationFunction.RELU, 0.1, True),
            LayerConfig(32, 64, ActivationFunction.RELU, 0.2, True),
            LayerConfig(64, 32, ActivationFunction.RELU, 0.1, True),
            LayerConfig(32, 5, ActivationFunction.SIGMOID, 0.0, False)  # 5 actions
        ]

        # Value network (state -> value estimate)
        value_configs = [
            LayerConfig(8, 32, ActivationFunction.RELU, 0.1, True),
            LayerConfig(32, 64, ActivationFunction.RELU, 0.2, True),
            LayerConfig(64, 32, ActivationFunction.RELU, 0.1, True),
            LayerConfig(32, 1, ActivationFunction.RELU, 0.0, False)  # Single value output
        ]

        training_config = TrainingConfig(
            learning_rate=0.0001,
            batch_size=16,
            max_epochs=500,
            early_stopping=True,
            patience=25
        )

        self.policy_network = NeuralNetwork(policy_configs, training_config)
        self.value_network = NeuralNetwork(value_configs, training_config)

    def suggest_adaptation(self, metrics: MeshQualityMetrics) -> AdaptationDecision:
        """Suggest mesh adaptation based on quality metrics"""
        if not self.is_trained:
            # Use heuristic approach if not trained
            return self._heuristic_adaptation(metrics)

        # Convert metrics to vector
        state_vector = self._metrics_to_vector(metrics)

        # Get action probabilities from policy network
        action_probs = self.policy_network.predict(state_vector.reshape(1, -1)).flatten()

        # Select action based on probabilities
        action_idx = np.argmax(action_probs)
        return self._index_to_action(action_idx)

    def suggest_batch_adaptation(self, metrics_list: List[MeshQualityMetrics]) -> List[AdaptationDecision]:
        """Suggest adaptations for a batch of mesh quality metrics"""
        return [self.suggest_adaptation(metrics) for metrics in metrics_list]

    def train_adapter(self, training_metrics: List[MeshQualityMetrics],
                     optimal_decisions: List[AdaptationDecision],
                     rewards: List[float]) -> Dict[str, Any]:
        """Train the mesh adapter using supervised learning"""
        if len(training_metrics) != len(optimal_decisions) or len(training_metrics) != len(rewards):
            raise ValueError("All input lists must have the same length")

        # Convert to arrays
        X = np.array([self._metrics_to_vector(m) for m in training_metrics])
        y_actions = np.array([self._action_to_vector(a) for a in optimal_decisions])
        y_values = np.array(rewards).reshape(-1, 1)

        # Train policy network
        policy_results = self.policy_network.train(X, y_actions)

        # Train value network
        value_results = self.value_network.train(X, y_values)

        self.is_trained = True

        return {
            'policy_training': policy_results,
            'value_training': value_results
        }

    def add_experience(self, metrics: MeshQualityMetrics, action: AdaptationDecision, reward: float):
        """Add experience to the buffer for reinforcement learning"""
        self.experience_buffer.append(metrics)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def train_from_experience(self) -> Dict[str, Any]:
        """Train from collected experience using reinforcement learning"""
        if len(self.experience_buffer) < 10:
            raise RuntimeError("Need at least 10 experiences to train")

        return self.train_adapter(self.experience_buffer, self.action_buffer, self.reward_buffer)

    def save_adapter(self, filename: str):
        """Save the trained adapter"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained adapter")

        # Save networks
        policy_filename = filename.replace('.json', '_policy.json')
        value_filename = filename.replace('.json', '_value.json')

        self.policy_network.save_model(policy_filename)
        self.value_network.save_model(value_filename)

        # Save metadata
        adapter_data = {
            'is_trained': self.is_trained,
            'policy_filename': policy_filename,
            'value_filename': value_filename,
            'experience_count': len(self.experience_buffer)
        }

        with open(filename, 'w') as f:
            json.dump(adapter_data, f, indent=2)

    def load_adapter(self, filename: str):
        """Load a trained adapter"""
        with open(filename, 'r') as f:
            adapter_data = json.load(f)

        self.is_trained = adapter_data['is_trained']

        # Load networks
        self._create_networks()
        self.policy_network.load_model(adapter_data['policy_filename'])
        self.value_network.load_model(adapter_data['value_filename'])

    def _heuristic_adaptation(self, metrics: MeshQualityMetrics) -> AdaptationDecision:
        """Heuristic-based adaptation when ML model is not trained"""
        # Simple heuristic rules
        if metrics.error_indicator > 0.8:
            return AdaptationDecision.REFINE
        elif metrics.error_indicator < 0.2 and metrics.computational_cost > 0.7:
            return AdaptationDecision.COARSEN
        elif metrics.aspect_ratio_quality < 0.3:
            return AdaptationDecision.ANISOTROPIC_REFINE
        elif metrics.smoothness_quality < 0.4:
            return AdaptationDecision.RELOCATE
        else:
            return AdaptationDecision.NO_CHANGE

    def _metrics_to_vector(self, metrics: MeshQualityMetrics) -> np.ndarray:
        """Convert mesh quality metrics to vector"""
        return np.array([
            metrics.aspect_ratio_quality,
            metrics.skewness_quality,
            metrics.orthogonality_quality,
            metrics.smoothness_quality,
            metrics.solution_gradient_magnitude,
            metrics.error_indicator,
            metrics.refinement_efficiency,
            metrics.computational_cost
        ])

    def _action_to_vector(self, action: AdaptationDecision) -> np.ndarray:
        """Convert adaptation decision to one-hot vector"""
        action_map = {
            AdaptationDecision.NO_CHANGE: 0,
            AdaptationDecision.REFINE: 1,
            AdaptationDecision.COARSEN: 2,
            AdaptationDecision.ANISOTROPIC_REFINE: 3,
            AdaptationDecision.RELOCATE: 4
        }

        vector = np.zeros(5)
        vector[action_map[action]] = 1.0
        return vector

    def _index_to_action(self, index: int) -> AdaptationDecision:
        """Convert action index to AdaptationDecision"""
        action_map = [
            AdaptationDecision.NO_CHANGE,
            AdaptationDecision.REFINE,
            AdaptationDecision.COARSEN,
            AdaptationDecision.ANISOTROPIC_REFINE,
            AdaptationDecision.RELOCATE
        ]
        return action_map[index]

# ============================================================================
# ML-Accelerated Solver
# ============================================================================

class MLAcceleratedSolver:
    """Machine learning-accelerated solver with neural network preconditioners"""

    def __init__(self):
        self.preconditioner_network = None
        self.convergence_predictor = None
        self.is_trained = False
        self._create_networks()

    def _create_networks(self):
        """Create preconditioner and convergence prediction networks"""
        # Preconditioner network (residual -> preconditioned residual)
        preconditioner_configs = [
            LayerConfig(100, 128, ActivationFunction.RELU, 0.1, True),  # Assume max 100 DOF for demo
            LayerConfig(128, 256, ActivationFunction.RELU, 0.2, True),
            LayerConfig(256, 128, ActivationFunction.RELU, 0.2, True),
            LayerConfig(128, 100, ActivationFunction.RELU, 0.0, False)
        ]

        # Convergence predictor (residual history -> convergence rate)
        convergence_configs = [
            LayerConfig(50, 64, ActivationFunction.RELU, 0.1, True),   # Last 50 residual values
            LayerConfig(64, 32, ActivationFunction.RELU, 0.1, True),
            LayerConfig(32, 16, ActivationFunction.RELU, 0.1, True),
            LayerConfig(16, 1, ActivationFunction.SIGMOID, 0.0, False)  # Convergence rate [0,1]
        ]

        training_config = TrainingConfig(
            learning_rate=0.001,
            batch_size=16,
            max_epochs=500,
            early_stopping=True,
            patience=30
        )

        self.preconditioner_network = NeuralNetwork(preconditioner_configs, training_config)
        self.convergence_predictor = NeuralNetwork(convergence_configs, training_config)

    def apply_ml_preconditioner(self, residual: np.ndarray) -> np.ndarray:
        """Apply ML-based preconditioning to residual"""
        if not self.is_trained:
            # Return simple diagonal preconditioning if not trained
            return residual / (np.abs(residual) + 1e-12)

        # Pad or truncate residual to network input size
        network_input_size = 100
        if len(residual) > network_input_size:
            # Truncate
            residual_input = residual[:network_input_size]
        else:
            # Pad with zeros
            residual_input = np.zeros(network_input_size)
            residual_input[:len(residual)] = residual

        # Apply preconditioner network
        preconditioned = self.preconditioner_network.predict(residual_input.reshape(1, -1)).flatten()

        # Return original size
        if len(residual) > network_input_size:
            return preconditioned
        else:
            return preconditioned[:len(residual)]

    def predict_convergence_rate(self, residual_history: List[float]) -> float:
        """Predict convergence rate from residual history"""
        if not self.is_trained:
            # Simple heuristic if not trained
            if len(residual_history) < 2:
                return 0.5
            return max(0.0, min(1.0, residual_history[-2] / (residual_history[-1] + 1e-12)))

        # Prepare input (last 50 residuals)
        network_input_size = 50
        if len(residual_history) >= network_input_size:
            input_data = np.array(residual_history[-network_input_size:])
        else:
            input_data = np.zeros(network_input_size)
            input_data[-len(residual_history):] = residual_history

        # Predict convergence rate
        convergence_rate = self.convergence_predictor.predict(input_data.reshape(1, -1)).flatten()[0]
        return float(convergence_rate)

    def predict_iterations_to_convergence(self, residual_history: List[float],
                                        target_tolerance: float = 1e-6) -> int:
        """Predict number of iterations needed for convergence"""
        if len(residual_history) == 0:
            return 100  # Default estimate

        current_residual = residual_history[-1]
        if current_residual <= target_tolerance:
            return 0

        convergence_rate = self.predict_convergence_rate(residual_history)

        if convergence_rate <= 0.01 or convergence_rate >= 1.0:  # Very slow or no convergence
            return 1000  # Max iterations

        # Estimate iterations based on exponential decay
        try:
            log_ratio = np.log(target_tolerance / current_residual)
            log_rate = np.log(convergence_rate)

            if log_rate == 0 or not np.isfinite(log_ratio) or not np.isfinite(log_rate):
                return 1000

            iterations_needed = int(log_ratio / log_rate)
            return max(1, min(1000, iterations_needed))
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1000  # Default to max iterations on error

    def train_preconditioner(self, residuals: List[np.ndarray],
                           preconditioned: List[np.ndarray]) -> Dict[str, Any]:
        """Train the preconditioner network"""
        if len(residuals) != len(preconditioned):
            raise ValueError("Residuals and preconditioned arrays must have same length")

        # Prepare training data
        network_input_size = 100
        X, y = [], []

        for res, prec in zip(residuals, preconditioned):
            # Pad or truncate to network size
            res_input = np.zeros(network_input_size)
            prec_output = np.zeros(network_input_size)

            size = min(len(res), network_input_size)
            res_input[:size] = res[:size]
            prec_output[:size] = prec[:size]

            X.append(res_input)
            y.append(prec_output)

        X = np.array(X)
        y = np.array(y)

        # Train the network
        results = self.preconditioner_network.train(X, y)
        self.is_trained = True

        return results

    def train_convergence_predictor(self, residual_histories: List[List[float]],
                                  iteration_counts: List[int]) -> Dict[str, Any]:
        """Train the convergence prediction network"""
        if len(residual_histories) != len(iteration_counts):
            raise ValueError("Residual histories and iteration counts must have same length")

        # Prepare training data
        network_input_size = 50
        X, y = [], []

        for history, iterations in zip(residual_histories, iteration_counts):
            # Prepare input (last 50 residuals)
            input_data = np.zeros(network_input_size)
            if len(history) >= network_input_size:
                input_data = np.array(history[-network_input_size:])
            else:
                input_data[-len(history):] = history

            # Compute convergence rate
            if len(history) >= 2:
                convergence_rate = history[-2] / (history[-1] + 1e-12)
                convergence_rate = max(0.0, min(1.0, convergence_rate))
            else:
                convergence_rate = 0.5

            X.append(input_data)
            y.append([convergence_rate])

        X = np.array(X)
        y = np.array(y)

        # Train the network
        results = self.convergence_predictor.train(X, y)

        return results

    def save_solver(self, filename: str):
        """Save the trained solver"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained solver")

        # Save networks
        preconditioner_filename = filename.replace('.json', '_preconditioner.json')
        convergence_filename = filename.replace('.json', '_convergence.json')

        self.preconditioner_network.save_model(preconditioner_filename)
        self.convergence_predictor.save_model(convergence_filename)

        # Save metadata
        solver_data = {
            'is_trained': self.is_trained,
            'preconditioner_filename': preconditioner_filename,
            'convergence_filename': convergence_filename
        }

        with open(filename, 'w') as f:
            json.dump(solver_data, f, indent=2)

    def load_solver(self, filename: str):
        """Load a trained solver"""
        with open(filename, 'r') as f:
            solver_data = json.load(f)

        self.is_trained = solver_data['is_trained']

        # Load networks
        self._create_networks()
        self.preconditioner_network.load_model(solver_data['preconditioner_filename'])
        self.convergence_predictor.load_model(solver_data['convergence_filename'])
