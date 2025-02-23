import sys
import json
import logging
import click
from src.data import read_data, split_train_validate_data, load_data_from_s3
from src.entities import TrainingPipelineParams, read_training_pipeline_params
from src.features import make_features, build_transformer, extract_target
from src.models import train_model, model_predict, evaluate_model, save_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f'Load data from minio...')
    load_data_from_s3(training_pipeline_params.s3_data_path, training_pipeline_params.output_data_folder)
    logger.info(f"Read data file...")
    df = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Data loaded successfully, total number of objects: {df.shape[0]}\n")
    logger.info(f"Split data to train and validate...")
    df_train, df_validate = split_train_validate_data(
        df, training_pipeline_params.splitting_params
    )
    logger.info(
        f"Train data size: {df_train.shape[0]}, validate data size: {df_validate.shape[0]}\n"
    )
    logger.info(f"Start build transformer for features...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(df_train)
    logger.info(f"Pipeline for features was built and fitted!!!\n")
    logger.info(f"Start make features...")
    train_features = make_features(df_train, transformer)
    validate_features = make_features(df_validate, transformer)
    logger.info(
        f"Prepared train and validate features, train features shape: {train_features.shape}, validate features shape: {validate_features.shape}\n"
    )
    train_target = extract_target(df_train, training_pipeline_params.feature_params)
    validate_target = extract_target(
        df_validate, training_pipeline_params.feature_params
    )
    logger.info(f"Start training model...")
    model = train_model(
        train_features, train_target, training_pipeline_params.training_params
    )
    logger.info(f"Model was trained successfully!!!\n")
    logger.info(f"Start getting metrics...!!!\n")
    predict = model_predict(validate_features, model)
    
    metrics = evaluate_model(predict, validate_target)
    with open(training_pipeline_params.metric_file_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(
        f"Metrics was saved in {training_pipeline_params.metric_file_path}!!!\n"
    )
    save_model(model, training_pipeline_params.local_model_save_path, training_pipeline_params.model_s3_path)
    logger.info(f"Model saved locally at {training_pipeline_params.local_model_save_path} and to minio: {training_pipeline_params.model_s3_path}!!!\n")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    train_pipeline(training_pipeline_params)


if __name__ == "__main__":
    train_pipeline_command()
