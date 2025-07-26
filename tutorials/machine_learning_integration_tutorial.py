"""
Machine Learning Integration Tutorial

This tutorial demonstrates the usage of machine learning capabilities in the SemiDGFEM simulator:
- Neural network-based device parameter prediction
- AI-enhanced mesh adaptation
- Machine learning-accelerated solvers
- Data-driven material property prediction

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from machine_learning_integration import (
    DeviceParameterPredictor, DeviceFeatures, PredictedParameters,
    AIEnhancedMeshAdapter, MeshQualityMetrics, AdaptationDecision,
    MLAcceleratedSolver, NeuralNetwork, LayerConfig, TrainingConfig,
    ActivationFunction, OptimizerType, LossFunction
)

def tutorial_1_device_parameter_prediction():
    """Tutorial 1: Neural Network-based Device Parameter Prediction"""
    print("\n" + "="*70)
    print("Tutorial 1: Neural Network-based Device Parameter Prediction")
    print("="*70)
    
    print("1.1 Creating device parameter predictor...")
    
    # Create predictor
    predictor = DeviceParameterPredictor()
    
    print("   - Neural network architecture:")
    print("     * Input layer: 16 features → 64 neurons (ReLU, BatchNorm, Dropout=0.1)")
    print("     * Hidden layer 1: 64 → 128 neurons (ReLU, BatchNorm, Dropout=0.2)")
    print("     * Hidden layer 2: 128 → 64 neurons (ReLU, BatchNorm, Dropout=0.2)")
    print("     * Hidden layer 3: 64 → 32 neurons (ReLU, BatchNorm, Dropout=0.1)")
    print("     * Output layer: 32 → 10 parameters (ReLU)")
    
    print("\n1.2 Generating synthetic training dataset...")
    
    # Generate synthetic training data for MOSFET devices
    np.random.seed(42)
    features_list = []
    parameters_list = []
    
    for i in range(500):
        # Generate realistic device features
        channel_length = np.random.uniform(10e-9, 100e-9)  # 10-100 nm
        channel_width = np.random.uniform(1e-6, 10e-6)     # 1-10 μm
        oxide_thickness = np.random.uniform(1e-9, 5e-9)    # 1-5 nm
        
        features = DeviceFeatures(
            channel_length=channel_length,
            channel_width=channel_width,
            oxide_thickness=oxide_thickness,
            junction_depth=np.random.uniform(10e-9, 50e-9),
            substrate_doping=np.random.uniform(1e15, 1e18),
            source_drain_doping=np.random.uniform(1e19, 1e21),
            gate_work_function=np.random.uniform(4.0, 5.0),
            oxide_permittivity=np.random.uniform(3.5, 4.0),
            temperature=np.random.uniform(250, 400),
            gate_voltage=np.random.uniform(0.0, 2.0),
            drain_voltage=np.random.uniform(0.0, 2.0),
            source_voltage=0.0,
            line_edge_roughness=np.random.uniform(0.0, 2e-9),
            oxide_thickness_variation=np.random.uniform(0.0, 0.1e-9),
            doping_fluctuation=np.random.uniform(0.0, 0.1),
            work_function_variation=np.random.uniform(0.0, 0.1)
        )
        
        # Generate realistic parameters with physics-based relationships
        # Threshold voltage depends on oxide thickness and work function
        vth_base = 0.3 + (oxide_thickness * 1e9) * 0.1 + (features.gate_work_function - 4.5) * 0.2
        threshold_voltage = vth_base + 0.05 * np.random.randn()
        
        # Subthreshold slope depends on temperature and oxide quality
        ss_base = 60 * (features.temperature / 300) * (1 + oxide_thickness * 1e10)
        subthreshold_slope = ss_base + 5 * np.random.randn()
        
        # Short channel effects depend on channel length
        sce_base = 0.1 * np.exp(-(channel_length * 1e9) / 20)
        short_channel_effect = sce_base + 0.02 * np.random.randn()
        
        parameters = PredictedParameters(
            threshold_voltage=threshold_voltage,
            subthreshold_slope=max(60, subthreshold_slope),
            drain_induced_barrier_lowering=short_channel_effect * 0.5,
            transconductance=channel_width / channel_length * 1e-6 * (1 + 0.1 * np.random.randn()),
            output_conductance=1e-5 * (1 + 0.2 * np.random.randn()),
            gate_leakage_current=1e-12 * np.exp(-oxide_thickness * 1e10) * (1 + 0.3 * np.random.randn()),
            junction_leakage_current=1e-15 * (1 + 0.2 * np.random.randn()),
            channel_mobility=300 * (300 / features.temperature) * (1 + 0.1 * np.random.randn()),
            saturation_velocity=1e5 * (1 + 0.05 * np.random.randn()),
            short_channel_effect=short_channel_effect
        )
        
        features_list.append(features)
        parameters_list.append(parameters)
    
    print(f"   - Generated {len(features_list)} device samples")
    print(f"   - Channel length range: {min(f.channel_length for f in features_list)*1e9:.1f} - {max(f.channel_length for f in features_list)*1e9:.1f} nm")
    print(f"   - Threshold voltage range: {min(p.threshold_voltage for p in parameters_list):.3f} - {max(p.threshold_voltage for p in parameters_list):.3f} V")
    
    print("\n1.3 Training the neural network predictor...")
    
    # Split data into training and test sets
    train_size = int(0.8 * len(features_list))
    train_features = features_list[:train_size]
    train_parameters = parameters_list[:train_size]
    test_features = features_list[train_size:]
    test_parameters = parameters_list[train_size:]
    
    # Train the predictor
    training_results = predictor.train_predictor(train_features, train_parameters)
    
    print(f"   - Training completed in {training_results['epochs_trained']} epochs")
    print(f"   - Final training loss: {training_results['final_training_loss']:.6f}")
    print(f"   - Final validation loss: {training_results['final_validation_loss']:.6f}")
    print(f"   - Converged: {training_results['converged']}")
    
    print("\n1.4 Validating predictor performance...")
    
    # Validate on test set
    validation_results = predictor.validate_predictor(test_features, test_parameters)
    
    print(f"   - Test set MSE: {validation_results['overall_mse']:.6f}")
    print(f"   - Test set MAE: {validation_results['overall_mae']:.6f}")
    print(f"   - Test set R²: {validation_results['overall_r2']:.4f}")
    
    # Show per-parameter performance
    print("   - Per-parameter MSE:")
    for param in ['threshold_voltage', 'subthreshold_slope', 'transconductance']:
        mse_key = f'{param}_mse'
        if mse_key in validation_results:
            print(f"     * {param}: {validation_results[mse_key]:.6f}")
    
    print("\n1.5 Making predictions on new devices...")
    
    # Create test devices with different characteristics
    test_devices = [
        DeviceFeatures(
            channel_length=20e-9, channel_width=5e-6, oxide_thickness=2e-9,
            junction_depth=30e-9, substrate_doping=5e16, source_drain_doping=5e20,
            gate_work_function=4.5, oxide_permittivity=3.9, temperature=300,
            gate_voltage=1.0, drain_voltage=1.0, source_voltage=0.0,
            line_edge_roughness=1e-9, oxide_thickness_variation=0.05e-9,
            doping_fluctuation=0.05, work_function_variation=0.05
        ),
        DeviceFeatures(
            channel_length=50e-9, channel_width=2e-6, oxide_thickness=3e-9,
            junction_depth=40e-9, substrate_doping=1e17, source_drain_doping=1e21,
            gate_work_function=4.2, oxide_permittivity=3.7, temperature=350,
            gate_voltage=1.5, drain_voltage=1.5, source_voltage=0.0,
            line_edge_roughness=0.5e-9, oxide_thickness_variation=0.02e-9,
            doping_fluctuation=0.02, work_function_variation=0.02
        )
    ]
    
    predictions = predictor.predict_batch(test_devices)
    
    for i, (device, pred) in enumerate(zip(test_devices, predictions)):
        print(f"   - Device {i+1} (L={device.channel_length*1e9:.0f}nm, W={device.channel_width*1e6:.0f}μm):")
        print(f"     * Threshold voltage: {pred.threshold_voltage:.3f} V")
        print(f"     * Subthreshold slope: {pred.subthreshold_slope:.1f} mV/dec")
        print(f"     * Transconductance: {pred.transconductance*1e6:.1f} μS")
        print(f"     * Channel mobility: {pred.channel_mobility:.0f} cm²/V·s")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training history
    plt.subplot(2, 3, 1)
    epochs = range(1, len(predictor.network.training_history) + 1)
    plt.semilogy(epochs, predictor.network.training_history, 'b-', label='Training Loss')
    plt.semilogy(epochs, predictor.network.validation_history, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Predicted vs Actual (Threshold Voltage)
    plt.subplot(2, 3, 2)
    actual_vth = [p.threshold_voltage for p in test_parameters]
    predicted_vth = [predictor.predict_parameters(f).threshold_voltage for f in test_features]
    plt.scatter(actual_vth, predicted_vth, alpha=0.6)
    plt.plot([min(actual_vth), max(actual_vth)], [min(actual_vth), max(actual_vth)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Threshold Voltage (V)')
    plt.ylabel('Predicted Threshold Voltage (V)')
    plt.title('Threshold Voltage Prediction')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Predicted vs Actual (Subthreshold Slope)
    plt.subplot(2, 3, 3)
    actual_ss = [p.subthreshold_slope for p in test_parameters]
    predicted_ss = [predictor.predict_parameters(f).subthreshold_slope for f in test_features]
    plt.scatter(actual_ss, predicted_ss, alpha=0.6)
    plt.plot([min(actual_ss), max(actual_ss)], [min(actual_ss), max(actual_ss)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Subthreshold Slope (mV/dec)')
    plt.ylabel('Predicted Subthreshold Slope (mV/dec)')
    plt.title('Subthreshold Slope Prediction')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Feature importance (channel length vs threshold voltage)
    plt.subplot(2, 3, 4)
    channel_lengths = [f.channel_length * 1e9 for f in test_features]
    plt.scatter(channel_lengths, predicted_vth, alpha=0.6, c='blue', label='Predicted')
    plt.scatter(channel_lengths, actual_vth, alpha=0.6, c='red', label='Actual')
    plt.xlabel('Channel Length (nm)')
    plt.ylabel('Threshold Voltage (V)')
    plt.title('Channel Length vs Threshold Voltage')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Oxide thickness vs gate leakage
    plt.subplot(2, 3, 5)
    oxide_thickness = [f.oxide_thickness * 1e9 for f in test_features]
    predicted_leakage = [predictor.predict_parameters(f).gate_leakage_current for f in test_features]
    actual_leakage = [p.gate_leakage_current for p in test_parameters]
    plt.semilogy(oxide_thickness, predicted_leakage, 'bo', alpha=0.6, label='Predicted')
    plt.semilogy(oxide_thickness, actual_leakage, 'ro', alpha=0.6, label='Actual')
    plt.xlabel('Oxide Thickness (nm)')
    plt.ylabel('Gate Leakage Current (A)')
    plt.title('Oxide Thickness vs Gate Leakage')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Prediction accuracy distribution
    plt.subplot(2, 3, 6)
    errors = np.array(predicted_vth) - np.array(actual_vth)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (V)')
    plt.ylabel('Frequency')
    plt.title('Threshold Voltage Prediction Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('device_parameter_prediction_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - Device parameter prediction visualization saved as 'device_parameter_prediction_tutorial.png'")

def tutorial_2_ai_enhanced_mesh_adaptation():
    """Tutorial 2: AI-Enhanced Mesh Adaptation"""
    print("\n" + "="*70)
    print("Tutorial 2: AI-Enhanced Mesh Adaptation")
    print("="*70)
    
    print("2.1 Creating AI-enhanced mesh adapter...")
    
    # Create adapter
    adapter = AIEnhancedMeshAdapter()
    
    print("   - Policy network architecture:")
    print("     * Input: 8 mesh quality metrics")
    print("     * Hidden layers: 32 → 64 → 32 neurons (ReLU, BatchNorm, Dropout)")
    print("     * Output: 5 adaptation decisions (Sigmoid)")
    print("   - Value network architecture:")
    print("     * Input: 8 mesh quality metrics")
    print("     * Hidden layers: 32 → 64 → 32 neurons (ReLU, BatchNorm, Dropout)")
    print("     * Output: 1 value estimate (ReLU)")
    
    print("\n2.2 Generating mesh adaptation training data...")
    
    # Generate synthetic mesh quality data
    np.random.seed(42)
    metrics_list = []
    decisions_list = []
    rewards_list = []
    
    for i in range(200):
        # Generate realistic mesh quality metrics
        aspect_ratio_quality = np.random.beta(2, 2)  # Biased towards middle values
        skewness_quality = np.random.beta(3, 2)      # Biased towards higher values
        orthogonality_quality = np.random.beta(3, 2)
        smoothness_quality = np.random.beta(2, 3)    # Biased towards lower values
        solution_gradient_magnitude = np.random.exponential(2.0)
        error_indicator = np.random.beta(2, 5)       # Biased towards lower values
        refinement_efficiency = np.random.beta(3, 2)
        computational_cost = np.random.beta(2, 3)
        
        metrics = MeshQualityMetrics(
            aspect_ratio_quality=aspect_ratio_quality,
            skewness_quality=skewness_quality,
            orthogonality_quality=orthogonality_quality,
            smoothness_quality=smoothness_quality,
            solution_gradient_magnitude=solution_gradient_magnitude,
            error_indicator=error_indicator,
            refinement_efficiency=refinement_efficiency,
            computational_cost=computational_cost
        )
        
        # Generate optimal decisions based on physics-informed heuristics
        if error_indicator > 0.7 and refinement_efficiency > 0.5:
            decision = AdaptationDecision.REFINE
            reward = 0.9 - computational_cost * 0.2
        elif error_indicator < 0.2 and computational_cost > 0.6:
            decision = AdaptationDecision.COARSEN
            reward = 0.7 + (1 - computational_cost) * 0.2
        elif aspect_ratio_quality < 0.3 and solution_gradient_magnitude > 3.0:
            decision = AdaptationDecision.ANISOTROPIC_REFINE
            reward = 0.8 - computational_cost * 0.1
        elif smoothness_quality < 0.4 and orthogonality_quality > 0.6:
            decision = AdaptationDecision.RELOCATE
            reward = 0.6 + smoothness_quality * 0.3
        else:
            decision = AdaptationDecision.NO_CHANGE
            reward = 0.5 + (1 - computational_cost) * 0.3
        
        # Add some noise to rewards
        reward += 0.1 * np.random.randn()
        reward = max(0.0, min(1.0, reward))
        
        metrics_list.append(metrics)
        decisions_list.append(decision)
        rewards_list.append(reward)
    
    print(f"   - Generated {len(metrics_list)} mesh adaptation samples")
    
    # Count decision types
    decision_counts = {}
    for decision in decisions_list:
        decision_counts[decision.value] = decision_counts.get(decision.value, 0) + 1
    
    print("   - Decision distribution:")
    for decision, count in decision_counts.items():
        print(f"     * {decision}: {count} ({count/len(decisions_list)*100:.1f}%)")
    
    print(f"   - Average reward: {np.mean(rewards_list):.3f} ± {np.std(rewards_list):.3f}")
    
    print("\n2.3 Training the AI mesh adapter...")
    
    # Split data for training and testing
    train_size = int(0.8 * len(metrics_list))
    train_metrics = metrics_list[:train_size]
    train_decisions = decisions_list[:train_size]
    train_rewards = rewards_list[:train_size]
    test_metrics = metrics_list[train_size:]
    test_decisions = decisions_list[train_size:]
    test_rewards = rewards_list[train_size:]
    
    # Train the adapter
    training_results = adapter.train_adapter(train_metrics, train_decisions, train_rewards)
    
    print(f"   - Policy network training:")
    print(f"     * Epochs: {training_results['policy_training']['epochs_trained']}")
    print(f"     * Final loss: {training_results['policy_training']['final_training_loss']:.6f}")
    print(f"   - Value network training:")
    print(f"     * Epochs: {training_results['value_training']['epochs_trained']}")
    print(f"     * Final loss: {training_results['value_training']['final_training_loss']:.6f}")
    
    print("\n2.4 Testing adaptation decisions...")
    
    # Test the trained adapter
    correct_predictions = 0
    for metrics, true_decision in zip(test_metrics, test_decisions):
        predicted_decision = adapter.suggest_adaptation(metrics)
        if predicted_decision == true_decision:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_metrics)
    print(f"   - Adaptation decision accuracy: {accuracy:.3f} ({correct_predictions}/{len(test_metrics)})")
    
    # Test specific scenarios
    print("\n2.5 Testing specific mesh scenarios...")
    
    scenarios = [
        ("High error, good efficiency", MeshQualityMetrics(0.8, 0.7, 0.8, 0.6, 5.0, 0.9, 0.8, 0.3)),
        ("Low error, high cost", MeshQualityMetrics(0.6, 0.8, 0.7, 0.5, 1.0, 0.1, 0.6, 0.9)),
        ("Poor aspect ratio", MeshQualityMetrics(0.2, 0.6, 0.7, 0.5, 4.0, 0.5, 0.7, 0.5)),
        ("Poor smoothness", MeshQualityMetrics(0.7, 0.8, 0.8, 0.2, 2.0, 0.4, 0.6, 0.4)),
        ("Good quality mesh", MeshQualityMetrics(0.9, 0.9, 0.9, 0.8, 1.0, 0.2, 0.8, 0.3))
    ]
    
    for scenario_name, metrics in scenarios:
        decision = adapter.suggest_adaptation(metrics)
        print(f"   - {scenario_name}: {decision.value}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training history for policy network
    plt.subplot(2, 3, 1)
    policy_epochs = range(1, len(adapter.policy_network.training_history) + 1)
    plt.semilogy(policy_epochs, adapter.policy_network.training_history, 'b-', label='Training Loss')
    plt.semilogy(policy_epochs, adapter.policy_network.validation_history, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Policy Network Training')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Decision distribution
    plt.subplot(2, 3, 2)
    decisions = list(decision_counts.keys())
    counts = list(decision_counts.values())
    plt.bar(decisions, counts)
    plt.xlabel('Adaptation Decision')
    plt.ylabel('Count')
    plt.title('Decision Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot 3: Error indicator vs decision
    plt.subplot(2, 3, 3)
    error_indicators = [m.error_indicator for m in metrics_list]
    decision_colors = {'refine': 'red', 'coarsen': 'blue', 'anisotropic_refine': 'green', 
                      'relocate': 'orange', 'no_change': 'gray'}
    
    for decision_type in decision_colors:
        indices = [i for i, d in enumerate(decisions_list) if d.value == decision_type]
        if indices:
            errors = [error_indicators[i] for i in indices]
            rewards = [rewards_list[i] for i in indices]
            plt.scatter(errors, rewards, c=decision_colors[decision_type], 
                       label=decision_type, alpha=0.6)
    
    plt.xlabel('Error Indicator')
    plt.ylabel('Reward')
    plt.title('Error Indicator vs Reward by Decision')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Computational cost vs decision
    plt.subplot(2, 3, 4)
    comp_costs = [m.computational_cost for m in metrics_list]
    
    for decision_type in decision_colors:
        indices = [i for i, d in enumerate(decisions_list) if d.value == decision_type]
        if indices:
            costs = [comp_costs[i] for i in indices]
            rewards = [rewards_list[i] for i in indices]
            plt.scatter(costs, rewards, c=decision_colors[decision_type], 
                       label=decision_type, alpha=0.6)
    
    plt.xlabel('Computational Cost')
    plt.ylabel('Reward')
    plt.title('Computational Cost vs Reward by Decision')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Mesh quality metrics correlation
    plt.subplot(2, 3, 5)
    aspect_ratios = [m.aspect_ratio_quality for m in metrics_list]
    skewness = [m.skewness_quality for m in metrics_list]
    plt.scatter(aspect_ratios, skewness, c=rewards_list, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Reward')
    plt.xlabel('Aspect Ratio Quality')
    plt.ylabel('Skewness Quality')
    plt.title('Mesh Quality Correlation')
    plt.grid(True)
    
    # Plot 6: Prediction accuracy by metric ranges
    plt.subplot(2, 3, 6)
    error_ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    accuracies = []
    
    for low, high in error_ranges:
        indices = [i for i, m in enumerate(test_metrics) 
                  if low <= m.error_indicator < high]
        if indices:
            correct = sum(1 for i in indices 
                         if adapter.suggest_adaptation(test_metrics[i]) == test_decisions[i])
            accuracy = correct / len(indices)
        else:
            accuracy = 0
        accuracies.append(accuracy)
    
    range_labels = [f'{low:.1f}-{high:.1f}' for low, high in error_ranges]
    plt.bar(range_labels, accuracies)
    plt.xlabel('Error Indicator Range')
    plt.ylabel('Prediction Accuracy')
    plt.title('Accuracy by Error Indicator Range')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ai_mesh_adaptation_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - AI mesh adaptation visualization saved as 'ai_mesh_adaptation_tutorial.png'")

def tutorial_3_ml_accelerated_solver():
    """Tutorial 3: Machine Learning-Accelerated Solver"""
    print("\n" + "="*70)
    print("Tutorial 3: Machine Learning-Accelerated Solver")
    print("="*70)

    print("3.1 Creating ML-accelerated solver...")

    # Create solver
    solver = MLAcceleratedSolver()

    print("   - Preconditioner network architecture:")
    print("     * Input: 100 residual components")
    print("     * Hidden layers: 128 → 256 → 128 neurons (ReLU, BatchNorm, Dropout)")
    print("     * Output: 100 preconditioned residual components")
    print("   - Convergence predictor architecture:")
    print("     * Input: 50 residual history values")
    print("     * Hidden layers: 64 → 32 → 16 neurons (ReLU, BatchNorm, Dropout)")
    print("     * Output: 1 convergence rate estimate (Sigmoid)")

    print("\n3.2 Generating solver training data...")

    # Generate synthetic linear system data
    np.random.seed(42)
    system_size = 50  # Smaller system for demonstration

    # Generate training data for preconditioner
    residuals_train = []
    preconditioned_train = []

    for i in range(100):
        # Generate random residual vector
        residual = np.random.randn(system_size)

        # Apply ideal diagonal preconditioning (ground truth)
        diagonal_elements = np.random.uniform(0.1, 2.0, system_size)
        preconditioned = residual / diagonal_elements

        residuals_train.append(residual)
        preconditioned_train.append(preconditioned)

    print(f"   - Generated {len(residuals_train)} preconditioner training samples")
    print(f"   - Residual vector size: {system_size}")

    # Generate training data for convergence predictor
    histories_train = []
    convergence_rates_train = []

    for i in range(100):
        # Generate synthetic convergence history
        initial_residual = np.random.uniform(1.0, 10.0)
        true_rate = np.random.uniform(0.1, 0.95)

        history = [initial_residual]
        for j in range(20):
            next_residual = history[-1] * true_rate * (1 + 0.05 * np.random.randn())
            history.append(max(1e-12, next_residual))

        histories_train.append(history)
        convergence_rates_train.append(len(history))

    print(f"   - Generated {len(histories_train)} convergence prediction samples")
    print(f"   - Average convergence history length: {np.mean([len(h) for h in histories_train]):.1f}")

    print("\n3.3 Training the ML solver components...")

    # Train preconditioner
    print("   - Training preconditioner network...")
    preconditioner_results = solver.train_preconditioner(residuals_train, preconditioned_train)

    print(f"     * Epochs: {preconditioner_results['epochs_trained']}")
    print(f"     * Final loss: {preconditioner_results['final_training_loss']:.6f}")
    print(f"     * Converged: {preconditioner_results['converged']}")

    # Train convergence predictor
    print("   - Training convergence predictor...")
    convergence_results = solver.train_convergence_predictor(histories_train, convergence_rates_train)

    print(f"     * Epochs: {convergence_results['epochs_trained']}")
    print(f"     * Final loss: {convergence_results['final_training_loss']:.6f}")
    print(f"     * Converged: {convergence_results['converged']}")

    print("\n3.4 Testing ML preconditioning...")

    # Test preconditioning on new residuals
    test_residuals = []
    test_preconditioned = []
    ml_preconditioned = []

    for i in range(10):
        residual = np.random.randn(system_size)
        diagonal_elements = np.random.uniform(0.1, 2.0, system_size)
        ideal_preconditioned = residual / diagonal_elements
        ml_result = solver.apply_ml_preconditioner(residual)

        test_residuals.append(residual)
        test_preconditioned.append(ideal_preconditioned)
        ml_preconditioned.append(ml_result)

    # Compute preconditioning accuracy
    mse_errors = []
    for ideal, ml_result in zip(test_preconditioned, ml_preconditioned):
        mse = np.mean((ideal - ml_result) ** 2)
        mse_errors.append(mse)

    print(f"   - Preconditioning MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
    print(f"   - Best case MSE: {np.min(mse_errors):.6f}")
    print(f"   - Worst case MSE: {np.max(mse_errors):.6f}")

    print("\n3.5 Testing convergence prediction...")

    # Test convergence prediction on new histories
    test_histories = []
    true_rates = []
    predicted_rates = []

    for i in range(20):
        # Generate test convergence history
        initial_residual = np.random.uniform(1.0, 5.0)
        true_rate = np.random.uniform(0.2, 0.9)

        history = [initial_residual]
        for j in range(15):
            next_residual = history[-1] * true_rate * (1 + 0.02 * np.random.randn())
            history.append(max(1e-12, next_residual))

        predicted_rate = solver.predict_convergence_rate(history)

        test_histories.append(history)
        true_rates.append(true_rate)
        predicted_rates.append(predicted_rate)

    # Compute prediction accuracy
    rate_errors = np.abs(np.array(true_rates) - np.array(predicted_rates))
    print(f"   - Convergence rate MAE: {np.mean(rate_errors):.4f} ± {np.std(rate_errors):.4f}")
    print(f"   - Best prediction error: {np.min(rate_errors):.4f}")
    print(f"   - Worst prediction error: {np.max(rate_errors):.4f}")

    # Test iteration prediction
    print("\n3.6 Testing iteration count prediction...")

    iteration_predictions = []
    for history in test_histories[:5]:
        predicted_iterations = solver.predict_iterations_to_convergence(history, 1e-6)
        iteration_predictions.append(predicted_iterations)

        print(f"   - History length {len(history)}, current residual {history[-1]:.2e}")
        print(f"     * Predicted iterations to 1e-6: {predicted_iterations}")

    # Demonstration of solver acceleration
    print("\n3.7 Demonstrating solver acceleration...")

    # Simulate iterative solver with and without ML acceleration
    def simulate_solver(use_ml=False, max_iterations=100):
        """Simulate an iterative solver"""
        residual_history = []
        current_residual = 1.0

        for iteration in range(max_iterations):
            residual_history.append(current_residual)

            if use_ml and len(residual_history) > 5:
                # Use ML to predict convergence and adjust step size
                predicted_rate = solver.predict_convergence_rate(residual_history)
                if predicted_rate < 0.1:  # Slow convergence detected
                    step_factor = 1.5  # Increase step size
                else:
                    step_factor = 1.0
            else:
                step_factor = 1.0

            # Simple convergence model
            base_rate = 0.7
            current_residual *= base_rate / step_factor

            # Add some noise
            current_residual *= (1 + 0.05 * np.random.randn())
            current_residual = max(1e-12, current_residual)

            if current_residual < 1e-6:
                break

        return residual_history

    # Compare standard vs ML-accelerated solver
    np.random.seed(42)
    standard_history = simulate_solver(use_ml=False)
    np.random.seed(42)
    ml_history = simulate_solver(use_ml=True)

    print(f"   - Standard solver iterations: {len(standard_history)}")
    print(f"   - ML-accelerated solver iterations: {len(ml_history)}")
    print(f"   - Speedup factor: {len(standard_history) / len(ml_history):.2f}x")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Preconditioner training history
    plt.subplot(2, 3, 1)
    prec_epochs = range(1, len(solver.preconditioner_network.training_history) + 1)
    plt.semilogy(prec_epochs, solver.preconditioner_network.training_history, 'b-', label='Training Loss')
    plt.semilogy(prec_epochs, solver.preconditioner_network.validation_history, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Preconditioner Training')
    plt.legend()
    plt.grid(True)

    # Plot 2: Convergence predictor training history
    plt.subplot(2, 3, 2)
    conv_epochs = range(1, len(solver.convergence_predictor.training_history) + 1)
    plt.semilogy(conv_epochs, solver.convergence_predictor.training_history, 'b-', label='Training Loss')
    plt.semilogy(conv_epochs, solver.convergence_predictor.validation_history, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Predictor Training')
    plt.legend()
    plt.grid(True)

    # Plot 3: Preconditioning accuracy
    plt.subplot(2, 3, 3)
    plt.scatter(range(len(mse_errors)), mse_errors, alpha=0.7)
    plt.axhline(y=np.mean(mse_errors), color='red', linestyle='--', label=f'Mean: {np.mean(mse_errors):.6f}')
    plt.xlabel('Test Case')
    plt.ylabel('MSE')
    plt.title('Preconditioning Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot 4: Convergence rate prediction
    plt.subplot(2, 3, 4)
    plt.scatter(true_rates, predicted_rates, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('True Convergence Rate')
    plt.ylabel('Predicted Convergence Rate')
    plt.title('Convergence Rate Prediction')
    plt.legend()
    plt.grid(True)

    # Plot 5: Solver comparison
    plt.subplot(2, 3, 5)
    plt.semilogy(range(len(standard_history)), standard_history, 'b-', label='Standard Solver', linewidth=2)
    plt.semilogy(range(len(ml_history)), ml_history, 'r-', label='ML-Accelerated Solver', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Solver Convergence Comparison')
    plt.legend()
    plt.grid(True)

    # Plot 6: Convergence rate distribution
    plt.subplot(2, 3, 6)
    plt.hist(predicted_rates, bins=15, alpha=0.7, label='Predicted Rates', edgecolor='black')
    plt.hist(true_rates, bins=15, alpha=0.7, label='True Rates', edgecolor='black')
    plt.xlabel('Convergence Rate')
    plt.ylabel('Frequency')
    plt.title('Convergence Rate Distribution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ml_accelerated_solver_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - ML-accelerated solver visualization saved as 'ml_accelerated_solver_tutorial.png'")

def run_all_tutorials():
    """Run all machine learning integration tutorials"""
    print("Machine Learning Integration Tutorial Suite")
    print("SemiDGFEM Semiconductor Device Simulator")
    print("Author: Dr. Mazharuddin Mohammed")

    try:
        tutorial_1_device_parameter_prediction()
        tutorial_2_ai_enhanced_mesh_adaptation()
        tutorial_3_ml_accelerated_solver()

        print("\n" + "="*70)
        print("✓ All tutorials completed successfully!")
        print("Generated visualization files:")
        print("  - device_parameter_prediction_tutorial.png")
        print("  - ai_mesh_adaptation_tutorial.png")
        print("  - ml_accelerated_solver_tutorial.png")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Tutorial failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tutorials()
