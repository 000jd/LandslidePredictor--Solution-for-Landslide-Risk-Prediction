import os
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"model_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_dummy_data(n_samples=100):
    """Generate dummy data with the same structure as the training data."""
    # Define the exact feature order expected by the models
    EXPECTED_FEATURES = [
        'elevation', 'slope', 'aspect', 'rainfall_daily', 'rainfall_monthly',
        'distance_to_faults', 'soil_depth', 'vegetation_density', 
        'earthquake_magnitude', 'soil_moisture', 'previous_landslides',
        'snow_melt', 'landslide_probability',
        'lithology_basalt', 'lithology_granite', 'lithology_limestone', 
        'lithology_sandstone', 'lithology_shale',
        'land_use_agriculture', 'land_use_barren', 'land_use_forest', 
        'land_use_grassland', 'land_use_urban',
        'human_activity_high', 'human_activity_low', 'human_activity_medium'
    ]

    # Generate numerical features
    np.random.seed(42)
    
    dummy_data = {
        'elevation': np.random.uniform(0, 3000, n_samples),
        'slope': np.random.uniform(0, 60, n_samples),
        'aspect': np.random.uniform(0, 360, n_samples),
        'rainfall_daily': np.random.uniform(0, 100, n_samples),
        'rainfall_monthly': np.random.uniform(0, 500, n_samples),
        'distance_to_faults': np.random.uniform(0, 5000, n_samples),
        'soil_depth': np.random.uniform(0, 5, n_samples),
        'vegetation_density': np.random.uniform(0, 1, n_samples),
        'earthquake_magnitude': np.random.uniform(0, 5, n_samples),
        'soil_moisture': np.random.uniform(0, 100, n_samples),
        'previous_landslides': np.random.randint(0, 10, n_samples),
        'snow_melt': np.random.uniform(0, 50, n_samples),
        'landslide_probability': np.random.uniform(0, 1, n_samples)
    }
    
    # Generate categorical features
    lithology_categories = ['basalt', 'granite', 'limestone', 'sandstone', 'shale']
    land_use_categories = ['agriculture', 'barren', 'forest', 'grassland', 'urban']
    human_activity_categories = ['high', 'low', 'medium']
    
    # Get random samples from each category (original method)
    lithology_samples = np.random.choice(lithology_categories, n_samples)
    land_use_samples = np.random.choice(land_use_categories, n_samples)
    human_activity_samples = np.random.choice(human_activity_categories, n_samples)
    
    # One-hot encode categorical variables
    # Original method: ensures only one category is 1 per sample
    for lithology in lithology_categories:
        dummy_data[f'lithology_{lithology}'] = (lithology_samples == lithology).astype(int)
    
    for land_use in land_use_categories:
        dummy_data[f'land_use_{land_use}'] = (land_use_samples == land_use).astype(int)
    
    for activity in human_activity_categories:
        dummy_data[f'human_activity_{activity}'] = (human_activity_samples == activity).astype(int)
    
    # Your fix method (commented out): random binary choice per category
    # Note: This may result in multiple 1s per sample, which might not match training
    # for lithology in lithology_categories:
    #     dummy_data[f'lithology_{lithology}'] = np.random.choice([0, 1], n_samples)
    # for land_use in land_use_categories:
    #     dummy_data[f'land_use_{land_use}'] = np.random.choice([0, 1], n_samples)
    # for activity in human_activity_categories:
    #     dummy_data[f'human_activity_{activity}'] = np.random.choice([0, 1], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(dummy_data)
    
    # Ensure all expected features are present
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with zeros
    
    # Reorder columns to match expected order
    return df[EXPECTED_FEATURES]

def load_models():
    """Load all models from saved_models directory."""
    models = {}
    model_dir = Path("saved_models")
    
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist!")
        return models
    
    for model_file in model_dir.glob("*.pkl"):
        try:
            model_name = model_file.stem.split('_')[0]  # Extract model name from filename
            logger.info(f"Loading model: {model_file}")
            model = joblib.load(model_file)
            
            # Handle models that support feature names validation
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model {model_file} supports feature name validation")
            
            models[str(model_file)] = {
                'instance': model,
                'features': getattr(model, 'feature_names_in_', None)
            }
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
    
    return models

def test_model(model, X_test):
    """Test a model and log the results."""
    try:
        # Validate feature names
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            if list(X_test.columns) != list(expected_features):
                raise ValueError(
                    f"Feature mismatch!\n"
                    f"Expected: {list(expected_features)}\n"
                    f"Actual: {list(X_test.columns)}"
                )
        
        # Get model prediction
        if hasattr(model, 'predict_proba'):
            try:
                # Try to get probability for positive class
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # Get probability for positive class
                logger.info(f"Model has predict_proba: Shape={y_pred_proba.shape}")
            except Exception as e:
                logger.warning(f"predict_proba failed, falling back to predict: {e}")
                y_pred = model.predict(X_test)
        else:
            # Use predict for models without predict_proba
            y_pred = model.predict(X_test)
            logger.info(f"Model has predict only: Shape={y_pred.shape}")
        
        # Get basic statistics
        if 'y_pred_proba' in locals():
            predictions = y_pred_proba
        else:
            predictions = y_pred
        
        stats = {
            "mean": np.mean(predictions),
            "min": np.min(predictions),
            "max": np.max(predictions),
            "std": np.std(predictions),
            "has_predict_proba": hasattr(model, 'predict_proba'),
            "number_of_samples": len(X_test)
        }
        
        logger.info(f"Model statistics: {stats}")
        
        # Check if predictions are reasonable
        if np.isnan(predictions).any():
            logger.error("Warning: Model produced NaN predictions")
        if np.isinf(predictions).any():
            logger.error("Warning: Model produced infinity predictions")
        
        return True, stats
    
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False, {"error": str(e)}

def main():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate dummy test data with correct feature order
    logger.info("Generating dummy test data...")
    X_test = generate_dummy_data(n_samples=100)
    logger.info(f"Generated data columns: {X_test.columns.tolist()}")
    
    # Load models
    logger.info("Loading models...")
    models = load_models()
    
    if not models:
        logger.error("No models found! Exiting...")
        return
    
    logger.info(f"Loaded {len(models)} models")
    
    # Test each model
    results = {}
    for model_path, model_info in models.items():
        logger.info(f"Testing model: {model_path}")
        model = model_info['instance']
        expected_features = model_info['features']
        
        # Validate features if available
        if expected_features is not None:
            logger.info(f"Model expects features: {expected_features}")
            try:
                X_test_reordered = X_test[expected_features]
            except KeyError as e:
                logger.error(f"Feature mismatch: {e}")
                continue
        else:
            X_test_reordered = X_test
            logger.warning("No feature names available for validation")
        
        # Perform the test with properly ordered features
        success, stats = test_model(model, X_test_reordered)
        results[model_path] = {
            "success": success,
            "stats": stats
        }
        
        if success:
            logger.info(f"Model {model_path} test successful")
        else:
            logger.error(f"Model {model_path} test failed")
    
    # Summary of results
    logger.info("=" * 50)
    logger.info("SUMMARY OF MODEL TESTS")
    logger.info("=" * 50)
    
    for model_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        logger.info(f"{model_name}: {status}")
        if result["success"]:
            logger.info(f"  - Mean prediction: {result['stats']['mean']:.4f}")
            logger.info(f"  - Range: [{result['stats']['min']:.4f} - {result['stats']['max']:.4f}]")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()