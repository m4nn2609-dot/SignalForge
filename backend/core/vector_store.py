import os
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class VectorStoreManager:
    """
    Milvus wrapper for Temporal RAG.
    Allows the Portfolio Strategist to query past failures without look-ahead bias.
    """
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self._connect()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
        except Exception as e:
            print(f"Milvus connection warning: {e}. Ensure Milvus is running.")

    def create_experiment_collection(self):
        """
        Creates a collection for storing experiment embeddings and metadata.
        """
        fields = [
            FieldSchema(name="exp_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536), # OpenAI/Llama standard
            FieldSchema(name="timestamp", dtype=DataType.INT64), # For Temporal Filtering
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="SignalForge Experiment Archive")
        return Collection("experiment_archive", schema)

    def add_experiment_to_rag(self, exp_id: str, embedding: List[float], timestamp: int, metadata: Dict[str, Any]):
        """
        Indexes an experiment for the Portfolio Strategist.
        """
        collection = Collection("experiment_archive")
        collection.insert([
            [exp_id],
            [embedding],
            [timestamp],
            [metadata]
        ])
        collection.flush()

    def query_past_failures(self, query_embedding: List[float], as_of_timestamp: int, limit: int = 5):
        """
        Temporal RAG: Find similar failed experiments that existed BEFORE as_of_timestamp.
        This physically prevents hindsight bias.
        """
        collection = Collection("experiment_archive")

        # Filter: timestamp < as_of_timestamp
        expr = f"timestamp < {as_of_timestamp}"

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
            expr=expr,
            output_fields=["metadata"]
        )
        return results
