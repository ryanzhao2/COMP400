"""
Run the LLM over dummy graph data to recommend HNSW parameters.
"""

import json
from vector_db_agent import (
    VectorDatabaseAgent,
    generate_dummy_graph_data,
)


def main():
    # Initialize agent (will use LLM if GEMINI_API_KEY is set via .env)
    agent = VectorDatabaseAgent()

    # Generate dummy graph data
    graph_data = generate_dummy_graph_data(dataset_size=4000, dimension=128, num_trials=6)

    # Ask for recommendation
    rec = agent.recommend_parameters_from_graph(graph_data)

    print("\n=== LLM Recommendation ===")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()


