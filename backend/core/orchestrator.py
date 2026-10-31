import operator
from typing import Annotated, Dict, Any, List, Union
from langgraph.graph import StateGraph, END
from backend.core.state import ResearchState
from backend.core.llm import LocalLLM
from backend.engine.quant_engine import QuantEngine
from backend.schemas.config import FeatureSuiteConfig, OperatorConfig

class SignalForgeOrchestrator:
    """
    The Decoupled Agentic Pipeline.
    Orchestrates the flow between AI Proposers and the Deterministic Engine.
    """
    def __init__(self):
        self.llm = LocalLLM()
        self.engine = QuantEngine()
        self.workflow = self._build_graph()

    def _build_graph(self):
        # 1. Initialize the Graph with our ResearchState
        workflow = StateGraph(ResearchState)

        # 2. Define the Nodes (The Agents)
        workflow.add_node("data_architect", self.node_data_architect)
        workflow.add_node("quant_analyst", self.node_quant_analyst)
        workflow.add_node("modeler", self.node_modeler)
        workflow.add_node("risk_auditor", self.node_risk_auditor)

        # 3. Define the Flow (Edges)
        workflow.set_entry_point("data_architect")

        # Sequence: Architect -> Analyst -> Modeler -> Auditor
        workflow.add_edge("data_architect", "quant_analyst")
        workflow.add_edge("quant_analyst", "modeler")
        workflow.add_edge("modeler", "risk_auditor")

        # Final step
        workflow.add_edge("risk_auditor", END)

        return workflow.compile()

    # --- Node Implementations ---

    def node_data_architect(self, state: ResearchState) -> Dict[str, Any]:
        """
        Agent: Data Architect
        Goal: Propose data cleaning and normalization strategies.
        """
        system_prompt = "You are a Data Architect. Analyze the OHLCV data and propose a cleaning JSON config."
        user_prompt = f"Symbol: {state['symbol']}, Timeframe: {state['timeframe']}. Propose cleaning steps."

        # AI proposes the config
        cleaning_config = self.llm.generate_json(system_prompt, user_prompt)

        # Note: In a full impl, we would call the engine here to actually clean the data
        return {"cleaning_config": cleaning_config, "iteration_count": state.get("iteration_count", 0) + 1}

    def node_quant_analyst(self, state: ResearchState) -> Dict[str, Any]:
        """
        Agent: Quant Analyst
        Goal: Propose a feature suite that captures alpha, then execute via QuantEngine.
        """
        system_prompt = (
            "You are a Quant Analyst. Propose a feature suite using only these operators: "
            "sma, ema, rsi, zscore, log_return. Output must be a valid FeatureSuiteConfig JSON."
        )
        user_prompt = f"Symbol: {state['symbol']}. Propose a high-predictive feature suite."

        # 1. AI Proposes JSON
        feature_json = self.llm.generate_json(system_prompt, user_prompt)

        # 2. Deterministic Validation (The Guardrail)
        try:
            config = FeatureSuiteConfig(**feature_json)
        except Exception as e:
            return {"errors": [f"Invalid Feature Config: {str(e)}"]}

        # 3. Deterministic Execution (The Engine)
        # Assuming state['cleaned_data'] was populated by the architect
        df = state.get("cleaned_data") or state.get("raw_data")
        if df is not None:
            processed_df = self.engine.execute_pipeline(df, config.features)

            # 4. Quant Rigor: Stationarity Check
            # Check the first feature as a sample
            sample_col = config.features[0].output_column or "feature_1"
            stationarity = self.engine.run_stationarity_test(processed_df[sample_col])

            return {
                "feature_config": feature_json,
                "processed_features": processed_df,
                "stationarity_report": stationarity
            }

        return {"errors": ["No data available for Quant Analyst"]}

    def node_modeler(self, state: ResearchState) -> Dict[str, Any]:
        """
        Agent: Modeler
        Goal: Implement Walk-Forward Validation to prevent overfitting.
        """
        # AI proposes the window sizes
        system_prompt = "You are a Modeler. Propose Walk-Forward Validation windows (train, val, step)."
        user_prompt = f"Data length: {len(state['processed_features'])} periods."

        val_config = self.llm.generate_json(system_prompt, user_prompt)

        return {"validation_config": val_config}

    def node_risk_auditor(self, state: ResearchState) -> Dict[str, Any]:
        """
        Agent: Risk Auditor (The Adversary)
        Goal: Calculate Sharpe, MaxDD and attempt to 'break' the signal.
        """
        # In a real scenario, this agent would run a backtest and analyze results
        system_prompt = "You are a Risk Auditor. Analyze the PnL and calculate Sharpe Ratio and Max Drawdown."
        user_prompt = "Check if the signal is overfitted or has excessive drawdown."

        verdict_json = self.llm.generate_json(system_prompt, user_prompt)

        return {
            "audit_verdict": verdict_json.get("verdict", "UNKNOWN"),
            "audit_reasoning": verdict_json.get("reasoning", "No reasoning provided")
        }

    def run_experiment(self, initial_state: ResearchState):
        """
        Executes the full LangGraph pipeline.
        """
        return self.workflow.invoke(initial_state)
