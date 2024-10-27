from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    """
    Read dataset from csv file and retrun pandas dataframe
    """
    df = pd.read_csv(path)
    return df


def split_train_validate_data(
    df: pd.DataFrame, splitting_params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe of all data to train and validate parts
    """
    df_train, df_validate = train_test_split(
        df,
        test_size=splitting_params.validate_size,
        random_state=splitting_params.random_state,
    )
    return df_train, df_validate
