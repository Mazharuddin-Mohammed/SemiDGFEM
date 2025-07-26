"""
Test Suite for Machine Learning Integration

This module tests the machine learning capabilities including:
- Neural network implementation
- Device parameter prediction
- AI-enhanced mesh adaptation
- ML-accelerated solvers

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from machine_learning_integration import (
    NeuralNetwork, NeuralLayer, LayerConfig, TrainingConfig,
    ActivationFunction, OptimizerType, LossFunction,
    DeviceParameterPredictor, DeviceFeatures, PredictedParameters,
    AIEnhancedMeshAdapter, MeshQualityMetrics, AdaptationDecision,
    MLAcceleratedSolver, StandardScaler
)

def test_neural_layer():
    """Test neural layer implementation"""
    print("Testing Neural Layer...")
    
    # Create layer configuration
    config = LayerConfig(
        input_size=10,
        output_size=5,
        activation=ActivationFunction.RELU,
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    # Create layer
    layer = NeuralLayer(config)
    
    # Test forward pass
    input_data = np.random.randn(3, 10)  # Batch of 3 samples
    output = layer.forward(input_data, training=True)
    
    assert output.shape == (3, 5), f"Expected output shape (3, 5), got {output.shape}"
    assert not np.any(np.isnan(output)), "Output contains NaN values"
    
    # Test parameter count
    param_count = layer.get_parameter_count()
    expected_count = 10 * 5 + 5 + 2 * 5  # weights + biases + batch_norm
    assert param_count == expected_count, f"Expected {expected_count} parameters, got {param_count}"
    
    print("✓ Neural Layer tests passed")

def test_neural_network():
    """Test neural network implementation"""
    print("Testing Neural Network...")
    
    # Create network configuration
    layer_configs = [
        LayerConfig(4, 8, ActivationFunction.RELU, 0.1, True),
        LayerConfig(8, 4, ActivationFunction.RELU, 0.1, True),
        LayerConfig(4, 2, ActivationFunction.SIGMOID, 0.0, False)
    ]
    
    training_config = TrainingConfig(
        learning_rate=0.01,
        batch_size=8,
        max_epochs=10,
        early_stopping=False
    )
    
    # Create network
    network = NeuralNetwork(layer_configs, training_config)
    
    # Test prediction
    input_data = np.random.randn(5, 4)
    predictions = network.predict(input_data)
    
    assert predictions.shape == (5, 2), f"Expected prediction shape (5, 2), got {predictions.shape}"
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
    
    # Test training
    X_train = np.random.randn(50, 4)
    y_train = np.random.randn(50, 2)
    
    training_results = network.train(X_train, y_train)
    
    assert 'final_training_loss' in training_results, "Training results missing final_training_loss"
    assert 'epochs_trained' in training_results, "Training results missing epochs_trained"
    
    # Test model saving and loading
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        model_file = f.name
    
    try:
        network.save_model(model_file)
        
        # Create new network and load
        new_network = NeuralNetwork(layer_configs, training_config)
        new_network.load_model(model_file)
        
        # Test that predictions are the same
        new_predictions = new_network.predict(input_data)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-10)
        
    finally:
        if os.path.exists(model_file):
            os.unlink(model_file)
    
    print("✓ Neural Network tests passed")

def test_device_parameter_predictor():
    """Test device parameter predictor"""
    print("Testing Device Parameter Predictor...")
    
    # Create predictor
    predictor = DeviceParameterPredictor()
    
    # Generate synthetic training data
    np.random.seed(42)
    features_list = []
    parameters_list = []
    
    for i in range(100):
        features = DeviceFeatures(
            channel_length=np.random.uniform(10e-9, 100e-9),
            channel_width=np.random.uniform(1e-6, 10e-6),
            oxide_thickness=np.random.uniform(1e-9, 5e-9),
            junction_depth=np.random.uniform(10e-9, 100e-9),
            substrate_doping=np.random.uniform(1e15, 1e18),
            source_drain_doping=np.random.uniform(1e19, 1e21),
            gate_work_function=np.random.uniform(4.0, 5.0),
            oxide_permittivity=np.random.uniform(3.0, 4.0),
            temperature=np.random.uniform(250, 400),
            gate_voltage=np.random.uniform(0.0, 2.0),
            drain_voltage=np.random.uniform(0.0, 2.0),
            source_voltage=0.0,
            line_edge_roughness=np.random.uniform(0.0, 2e-9),
            oxide_thickness_variation=np.random.uniform(0.0, 0.1e-9),
            doping_fluctuation=np.random.uniform(0.0, 0.1),
            work_function_variation=np.random.uniform(0.0, 0.1)
        )
        
        # Generate synthetic parameters (simplified relationships)
        parameters = PredictedParameters(
            threshold_voltage=0.5 + 0.1 * np.random.randn(),
            subthreshold_slope=60 + 10 * np.random.randn(),
            drain_induced_barrier_lowering=0.1 + 0.05 * np.random.randn(),
            transconductance=1e-3 + 0.1e-3 * np.random.randn(),
            output_conductance=1e-5 + 0.1e-5 * np.random.randn(),
            gate_leakage_current=1e-12 + 0.1e-12 * np.random.randn(),
            junction_leakage_current=1e-15 + 0.1e-15 * np.random.randn(),
            channel_mobility=300 + 50 * np.random.randn(),
            saturation_velocity=1e5 + 0.1e5 * np.random.randn(),
            short_channel_effect=0.1 + 0.05 * np.random.randn()
        )
        
        features_list.append(features)
        parameters_list.append(parameters)
    
    # Train predictor
    training_results = predictor.train_predictor(features_list[:80], parameters_list[:80])
    
    assert predictor.is_trained, "Predictor should be trained"
    assert 'final_training_loss' in training_results, "Training results missing final_training_loss"
    
    # Test prediction
    test_features = features_list[80]
    predicted_params = predictor.predict_parameters(test_features)
    
    assert isinstance(predicted_params, PredictedParameters), "Prediction should return PredictedParameters"
    
    # Test batch prediction
    batch_predictions = predictor.predict_batch(features_list[80:85])
    assert len(batch_predictions) == 5, "Batch prediction should return 5 results"
    
    # Test validation
    validation_results = predictor.validate_predictor(features_list[80:], parameters_list[80:])
    
    assert 'overall_mse' in validation_results, "Validation results missing overall_mse"
    assert 'overall_r2' in validation_results, "Validation results missing overall_r2"
    
    # Test model saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        model_file = os.path.join(temp_dir, 'predictor.pkl')
        
        predictor.save_predictor(model_file)
        
        # Create new predictor and load
        new_predictor = DeviceParameterPredictor()
        new_predictor.load_predictor(model_file)
        
        # Test that predictions are similar
        new_predicted = new_predictor.predict_parameters(test_features)
        
        # Check that the predictions are reasonably close
        assert abs(predicted_params.threshold_voltage - new_predicted.threshold_voltage) < 0.1
    
    print("✓ Device Parameter Predictor tests passed")

def test_ai_enhanced_mesh_adapter():
    """Test AI-enhanced mesh adapter"""
    print("Testing AI-Enhanced Mesh Adapter...")
    
    # Create adapter
    adapter = AIEnhancedMeshAdapter()
    
    # Generate synthetic training data
    np.random.seed(42)
    metrics_list = []
    decisions_list = []
    rewards_list = []
    
    for i in range(50):
        metrics = MeshQualityMetrics(
            aspect_ratio_quality=np.random.uniform(0.0, 1.0),
            skewness_quality=np.random.uniform(0.0, 1.0),
            orthogonality_quality=np.random.uniform(0.0, 1.0),
            smoothness_quality=np.random.uniform(0.0, 1.0),
            solution_gradient_magnitude=np.random.uniform(0.0, 10.0),
            error_indicator=np.random.uniform(0.0, 1.0),
            refinement_efficiency=np.random.uniform(0.0, 1.0),
            computational_cost=np.random.uniform(0.0, 1.0)
        )
        
        # Generate decision based on simple heuristic
        if metrics.error_indicator > 0.7:
            decision = AdaptationDecision.REFINE
            reward = 0.8
        elif metrics.error_indicator < 0.3:
            decision = AdaptationDecision.COARSEN
            reward = 0.6
        else:
            decision = AdaptationDecision.NO_CHANGE
            reward = 0.5
        
        metrics_list.append(metrics)
        decisions_list.append(decision)
        rewards_list.append(reward)
    
    # Test heuristic adaptation (before training)
    test_metrics = metrics_list[0]
    decision = adapter.suggest_adaptation(test_metrics)
    assert isinstance(decision, AdaptationDecision), "Should return AdaptationDecision"
    
    # Train adapter
    training_results = adapter.train_adapter(metrics_list[:40], decisions_list[:40], rewards_list[:40])
    
    assert adapter.is_trained, "Adapter should be trained"
    assert 'policy_training' in training_results, "Training results missing policy_training"
    assert 'value_training' in training_results, "Training results missing value_training"
    
    # Test trained adaptation
    trained_decision = adapter.suggest_adaptation(test_metrics)
    assert isinstance(trained_decision, AdaptationDecision), "Should return AdaptationDecision"
    
    # Test batch adaptation
    batch_decisions = adapter.suggest_batch_adaptation(metrics_list[40:45])
    assert len(batch_decisions) == 5, "Batch adaptation should return 5 decisions"
    
    # Test experience collection
    adapter.add_experience(test_metrics, decision, 0.7)
    assert len(adapter.experience_buffer) == 1, "Experience buffer should have 1 entry"
    
    # Test model saving and loading
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        model_file = f.name
    
    try:
        adapter.save_adapter(model_file)
        
        # Create new adapter and load
        new_adapter = AIEnhancedMeshAdapter()
        new_adapter.load_adapter(model_file)
        
        assert new_adapter.is_trained, "Loaded adapter should be trained"
        
    finally:
        # Clean up files
        for suffix in ['', '_policy.json', '_value.json']:
            file_to_remove = model_file.replace('.json', suffix + '.json') if suffix else model_file
            if os.path.exists(file_to_remove):
                os.unlink(file_to_remove)
    
    print("✓ AI-Enhanced Mesh Adapter tests passed")

def test_ml_accelerated_solver():
    """Test ML-accelerated solver"""
    print("Testing ML-Accelerated Solver...")
    
    # Create solver
    solver = MLAcceleratedSolver()
    
    # Test untrained preconditioning
    residual = np.random.randn(50)
    preconditioned = solver.apply_ml_preconditioner(residual)
    
    assert preconditioned.shape == residual.shape, "Preconditioned residual should have same shape"
    assert not np.any(np.isnan(preconditioned)), "Preconditioned residual should not contain NaN"
    
    # Test convergence prediction
    residual_history = [1.0, 0.5, 0.25, 0.125, 0.0625]
    convergence_rate = solver.predict_convergence_rate(residual_history)
    
    assert 0.0 <= convergence_rate <= 1.0, f"Convergence rate should be in [0,1], got {convergence_rate}"
    
    iterations_needed = solver.predict_iterations_to_convergence(residual_history, 1e-6)
    assert iterations_needed > 0, "Iterations needed should be positive"
    
    # Generate training data for preconditioner
    residuals_train = []
    preconditioned_train = []
    
    for i in range(20):
        res = np.random.randn(50)
        # Simple diagonal preconditioning as ground truth
        prec = res / (np.abs(res) + 0.1)
        
        residuals_train.append(res)
        preconditioned_train.append(prec)
    
    # Train preconditioner
    preconditioner_results = solver.train_preconditioner(residuals_train, preconditioned_train)
    
    assert 'final_training_loss' in preconditioner_results, "Training results missing final_training_loss"
    
    # Generate training data for convergence predictor
    histories_train = []
    iterations_train = []
    
    for i in range(20):
        # Generate synthetic convergence history
        initial_residual = np.random.uniform(1.0, 10.0)
        rate = np.random.uniform(0.1, 0.9)
        
        history = [initial_residual]
        for j in range(10):
            history.append(history[-1] * rate)
        
        histories_train.append(history)
        iterations_train.append(len(history))
    
    # Train convergence predictor
    convergence_results = solver.train_convergence_predictor(histories_train, iterations_train)
    
    assert 'final_training_loss' in convergence_results, "Training results missing final_training_loss"
    assert solver.is_trained, "Solver should be trained"
    
    # Test trained preconditioning
    trained_preconditioned = solver.apply_ml_preconditioner(residual)
    assert trained_preconditioned.shape == residual.shape, "Trained preconditioned residual should have same shape"
    
    # Test model saving and loading
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        model_file = f.name
    
    try:
        solver.save_solver(model_file)
        
        # Create new solver and load
        new_solver = MLAcceleratedSolver()
        new_solver.load_solver(model_file)
        
        assert new_solver.is_trained, "Loaded solver should be trained"
        
    finally:
        # Clean up files
        for suffix in ['', '_preconditioner.json', '_convergence.json']:
            file_to_remove = model_file.replace('.json', suffix + '.json') if suffix else model_file
            if os.path.exists(file_to_remove):
                os.unlink(file_to_remove)
    
    print("✓ ML-Accelerated Solver tests passed")

def test_standard_scaler():
    """Test standard scaler implementation"""
    print("Testing Standard Scaler...")
    
    # Create test data
    X = np.random.randn(100, 5) * 10 + 5  # Mean=5, std=10
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check that scaled data has zero mean and unit variance
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
    
    # Test inverse transform
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X, X_recovered, rtol=1e-10)
    
    print("✓ Standard Scaler tests passed")

def run_all_tests():
    """Run all machine learning integration tests"""
    print("Machine Learning Integration Test Suite")
    print("=" * 50)
    
    try:
        test_neural_layer()
        test_neural_network()
        test_device_parameter_predictor()
        test_ai_enhanced_mesh_adapter()
        test_ml_accelerated_solver()
        test_standard_scaler()
        
        print("\n" + "=" * 50)
        print("✓ All Machine Learning Integration tests passed successfully!")
        print("Total test categories: 6")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
