from typing import TypedDict, Dict, Any, List, Optional
import pandas as pd

class ResearchState(TypedDict):
    """
    The global state for a research experiment.
    Passed between agents in the LangGraph pipeline.
    """
    # Metadata
    symbol: str
    timeframe: str
    experiment_id: str

    # Data evolution
    raw_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    processed_features: Optional[pd.DataFrame]

    # Configuration (The JSON Contract)
    cleaning_config: Optional[Dict[str, Any]]
    feature_config: Optional[Dict[str, Any]]
    validation_config: Optional[Dict[str, Any]]

    # Results & Rigor
    stationarity_report: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    audit_verdict: Optional[str] # "APPROVED" or "REJECTED"
    audit_reasoning: Optional[str]

    # Iteration control
    errors: List[str]
    iteration_count: int
