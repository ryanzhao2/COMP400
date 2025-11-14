from typing import Dict, Any, List
from .models import GraphData, DistanceStats, TrialRecord
import json


def graph_data_to_dict(graph: GraphData) -> Dict[str, Any]:
    return {
        "dataset_size": graph.dataset_size,
        "dimension": graph.dimension,
        "global_distance_stats": {
            "mean": graph.global_distance_stats.mean,
            "std": graph.global_distance_stats.std,
            "minimum": graph.global_distance_stats.minimum,
            "maximum": graph.global_distance_stats.maximum,
            "p25": graph.global_distance_stats.p25,
            "p50": graph.global_distance_stats.p50,
            "p75": graph.global_distance_stats.p75,
        },
        "trials": [
            {
                "params": t.params,
                "metrics": t.metrics,
                "distance_stats": {
                    "mean": t.distance_stats.mean,
                    "std": t.distance_stats.std,
                    "minimum": t.distance_stats.minimum,
                    "maximum": t.distance_stats.maximum,
                    "p25": t.distance_stats.p25,
                    "p50": t.distance_stats.p50,
                    "p75": t.distance_stats.p75,
                },
            }
            for t in graph.trials
        ],
    }


def graph_data_from_dict(d: Dict[str, Any]) -> GraphData:
    gstats = d.get("global_distance_stats", {})
    global_stats = DistanceStats(
        mean=float(gstats.get("mean", 0.0)),
        std=float(gstats.get("std", 0.0)),
        minimum=float(gstats.get("minimum", 0.0)),
        maximum=float(gstats.get("maximum", 0.0)),
        p25=float(gstats.get("p25", 0.0)),
        p50=float(gstats.get("p50", 0.0)),
        p75=float(gstats.get("p75", 0.0)),
    )

    trials: List[TrialRecord] = []
    for td in d.get("trials", []):
        dstats_raw = td.get("distance_stats", {})
        dstats = DistanceStats(
            mean=float(dstats_raw.get("mean", 0.0)),
            std=float(dstats_raw.get("std", 0.0)),
            minimum=float(dstats_raw.get("minimum", 0.0)),
            maximum=float(dstats_raw.get("maximum", 0.0)),
            p25=float(dstats_raw.get("p25", 0.0)),
            p50=float(dstats_raw.get("p50", 0.0)),
            p75=float(dstats_raw.get("p75", 0.0)),
        )
        trials.append(
            TrialRecord(
                params={k: int(v) for k, v in td.get("params", {}).items()},
                metrics={k: float(v) for k, v in td.get("metrics", {}).items()},
                distance_stats=dstats,
            )
        )

    return GraphData(
        dataset_size=int(d.get("dataset_size", 0)),
        dimension=int(d.get("dimension", 0)),
        global_distance_stats=global_stats,
        trials=trials,
    )


def save_graph_data_json(graph: GraphData, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data_to_dict(graph), f, indent=2)


def load_graph_data_json(path: str) -> GraphData:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return graph_data_from_dict(data)


