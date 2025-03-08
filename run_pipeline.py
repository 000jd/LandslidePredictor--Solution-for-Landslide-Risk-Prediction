import argparse

from src.config.logging_config import setup_logging
from src.config.model_config import ModelNameConfig
from src.data.data_loader import ingest_data
from src.data.data_pipeline import clean_data

from src.train import train_model
from src.evaluation import MSE, R2Score, RMSE
from src.utils import check_data, save_model

logger = setup_logging()

def main(model_name, fine_tuning):
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_data = ingest_data("/home/joydip/Documents/Devlopment/LandslidePredictor---End-to-End-MLOps-Solution-for-Landslide-Risk-Prediction/data/synthetic_landslide_data_2.csv")
        
        # Print raw data info
        logger.info(f"Raw data shape: {raw_data.shape}")
        logger.info(f"Raw data columns: {raw_data.columns.tolist()}")
        logger.info(f"Raw data types:\n{raw_data.dtypes}")
        
        x_train, x_test, y_train, y_test = clean_data(raw_data)

        # Check data for issues
        check_data(x_train, y_train, "training")
        check_data(x_test, y_test, "testing")

        # Set up model configuration
        logger.info(f"Setting up {model_name} model with fine_tuning={fine_tuning}")
        model_config = ModelNameConfig(model_name=model_name, fine_tuning=fine_tuning)

        # Train the model
        logger.info("Training the model...")
        trained_model = train_model(x_train, x_test, y_train, y_test, model_config)

        # Save the model
        save_model(trained_model, model_name)

        # Evaluate the model
        logger.info("Evaluating the model...")
        y_pred = trained_model.predict(x_test)
        mse = MSE().calculate_score(y_test, y_pred)
        r2 = R2Score().calculate_score(y_test, y_pred)
        rmse = RMSE().calculate_score(y_test, y_pred)

        logger.info(f"Model evaluation results:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")

        logger.info("Model training and evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception("Traceback:")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a landslide prediction model")
    parser.add_argument("--model", type=str, default="randomforest", choices=["randomforest", "xgboost", "lightgbm", "linear_regression"], help="Model to train")
    parser.add_argument("--train", action="store_true", help="Enable with hyperparameter fine-tuning")
    args = parser.parse_args()

    main(args.model, args.train)