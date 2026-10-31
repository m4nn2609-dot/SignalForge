from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid

from backend.core.orchestrator import SignalForgeOrchestrator
from backend.core.state import ResearchState
from backend.core.db import DatabaseManager

app = FastAPI(title="SignalForge API", description="Professional-grade Quant Research Platform")

# Initialize Core Systems
orchestrator = SignalForgeOrchestrator()
db = DatabaseManager()

class ExperimentRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    metadata: Optional[Dict[str, Any]] = None

class ExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    message: str

@app.get("/health")
async def health_check():
    return {"status": "online", "engine": "deterministic", "orchestrator": "langgraph"}

@app.post("/research/start", response_model=ExperimentResponse)
async def start_research(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """
    Triggers the Decoupled Agentic Pipeline.
    Runs the LangGraph orchestrator in the background.
    """
    experiment_id = str(uuid.uuid4())

    # Initial state for the LangGraph pipeline
    initial_state: ResearchState = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "experiment_id": experiment_id,
        "raw_data": None, # In real impl, this would be fetched from a data provider
        "cleaned_data": None,
        "processed_features": None,
        "cleaning_config": None,
        "feature_config": None,
        "validation_config": None,
        "stationarity_report": None,
        "performance_metrics": None,
        "audit_verdict": None,
        "audit_reasoning": None,
        "errors": [],
        "iteration_count": 0
    }

    # Run the orchestrator as a background task to avoid blocking the API
    background_tasks.add_task(run_pipeline_and_save, initial_state)

    return {
        "experiment_id": experiment_id,
        "status": "started",
        "message": f"Research pipeline triggered for {request.symbol}"
    }

@app.get("/research/status/{experiment_id}")
async def get_status(experiment_id: str):
    """
    Fetches the current status and results of an experiment from the immutable DB.
    """
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": exp.id,
        "symbol": exp.symbol,
        "verdict": exp.audit_verdict,
        "sharpe": exp.sharpe_ratio,
        "drawdown": exp.max_drawdown,
        "reasoning": exp.audit_reasoning,
        "config": exp.feature_config
    }

async def run_pipeline_and_save(state: ResearchState):
    """
    Helper function to execute the LangGraph pipeline and persist results to the DB.
    """
    try:
        # 1. Execute the LangGraph pipeline
        final_state = orchestrator.run_experiment(state)

        # 2. Save the result to the Immutable Database
        db.save_experiment({
            "id": final_state["experiment_id"],
            "symbol": final_state["symbol"],
            "timeframe": final_state["timeframe"],
            "feature_config": final_state.get("feature_config"),
            "validation_config": final_state.get("validation_config"),
            "cleaning_config": final_state.get("cleaning_config"),
            "sharpe_ratio": final_state.get("performance_metrics", {}).get("sharpe"),
            "max_drawdown": final_state.get("performance_metrics", {}).get("max_dd"),
            "audit_verdict": final_state.get("audit_verdict"),
            "audit_reasoning": final_state.get("audit_reasoning")
        })
    except Exception as e:
        print(f"Pipeline Error for {state['experiment_id']}: {str(e)}")
