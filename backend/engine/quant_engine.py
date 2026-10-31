import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, List
from backend.schemas.config import OperatorConfig, OperatorType

class QuantEngine:
    """
    The Deterministic Execution Engine.
    It takes a JSON config and a DataFrame, and produces features.
    NO LLM logic exists here.
    """
    def __init__(self):
        # Map OperatorType to actual Python functions
        self._operators = {
            OperatorType.SMA: self._compute_sma,
            OperatorType.EMA: self._compute_ema,
            OperatorType.RSI: self._compute_rsi,
            OperatorType.ZSCORE: self._compute_zscore,
            OperatorType.LOG_RETURN: self._compute_log_return,
        }

    def execute_pipeline(self, df: pd.DataFrame, configs: List[OperatorConfig]) -> pd.DataFrame:
        """
        Executes a sequence of operations on the dataframe.
        """
        working_df = df.copy()

        for config in configs:
            if config.operator not in self._operators:
                raise NotImplementedError(f"Operator {config.operator} not implemented in Engine.")

            op_func = self._operators[config.operator]
            out_col = config.output_column or f"{config.input_column}_{config.operator.value}"

            working_df[out_col] = op_func(working_df[config.input_column], **config.params)

        return working_df

    # --- Deterministic Math Implementations ---

    def _compute_sma(self, series: pd.Series, window: int = 20) -> pd.Series:
        return series.rolling(window=window).mean()

    def _compute_ema(self, series: pd.Series, window: int = 20) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()

    def _compute_log_return(self, series: pd.Series) -> pd.Series:
        return np.log(series / series.shift(1))

    def run_stationarity_test(self, series: pd.Series) -> Dict[str, Any]:
        """
        ADF test to ensure the feature is stationary.
        Crucial for preventing spurious regressions.
        """
        # Drop NaNs for the test as adfuller doesn't accept them
        clean_series = series.dropna()
        if len(clean_series) < 20:
            return {"error": "Insufficient data for ADF test"}

        result = adfuller(clean_series)
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05
        }
