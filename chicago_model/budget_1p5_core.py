from __future__ import annotations

import gc
import os
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


try:
    from numba import njit as _numba_njit
    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = None
except Exception as exc:  
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = exc

    def _numba_njit(*decorator_args, **decorator_kwargs):

        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def _identity_decorator(func):
            return func

        return _identity_decorator

njit = _numba_njit

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

MILES_TO_KM = 1.609344
ALGORITHM_EPS_TIME_MIN = 1.0e-6
ALGORITHM_EPS_DISTANCE_MILE = 1.0e-9
INF = float("inf")


CAPACITY_PER_LANE_VEHPH = 1900.0
MAX_INFERRED_LANES = 5


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    scenario_label: str
    use_flood_road_state: bool
    operational_stations: Tuple[str, ...]
    analysis_hours: float = 3.0
    battery_range_miles: float = 235.0
    initial_soc_fractions: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    ev_class_shares: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    theta_ev: float = 0.2
    theta_non_ev: float = 0.2
    alpha_bpr: float = 0.15
    beta_bpr: float = 4.0
    max_msa_iterations: int = 1200
    min_msa_iterations: int = 10
    msa_tolerance: float = 1.0e-4
    demand_scale: float = 1.0
    print_all_links: bool = True
    progress_every: int = 10


    queue_wait_disutility_multiplier: float = 2.0
    queue_utilization_penalty_start: float = 0.85
    queue_utilization_penalty_scale_min: float = 5.0
    queue_utilization_penalty_power: float = 1.0
    queue_utilization_penalty_cap_min: float = 1.0e6
    queue_unstable_loading_penalty_min: float = 1.0e6

    ev_initial_k_paths: int = 5
    ev_path_refresh_every: int = 10
    ev_prefer_distinct_stations: bool = True
    ev_path_minhash_size: int = 8
    ev_path_diversity_weight: float = 0.80
    ev_path_cost_weight: float = 0.20
    ev_path_random_seed: int = 20260817


    ev_priority_station_id: str = "C6"
    ev_require_priority_station_if_feasible: bool = True


    ev_direct_candidate_trees: int = 8
    ev_direct_perturbation_strength: float = 0.35

    ev_yen_search_k: int = 5
    ev_yen_n_jobs: int = -1
    ev_yen_rebuild_cache: bool = False
    ev_yen_progress_every: int = 5


@dataclass
class NetworkData:
    links: pd.DataFrame
    stations: pd.DataFrame
    node_ids: np.ndarray
    node_to_index: Dict[int, int]
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_type: np.ndarray
    edge_length_miles: np.ndarray
    edge_t0_dry_min: np.ndarray
    edge_t0_flood_min: np.ndarray
    edge_capacity_dry: np.ndarray
    edge_capacity_flood: np.ndarray
    edge_is_closed: np.ndarray
    edge_station_id: np.ndarray
    active_edge: np.ndarray
    travel_edge: np.ndarray
    station_ids: List[str]
    station_ports: np.ndarray
    station_service_min: np.ndarray
    station_entry_edge: np.ndarray
    station_exit_edge: np.ndarray
    station_bypass_edge: np.ndarray
    station_entrance_node: np.ndarray
    station_exit_node: np.ndarray
    operational_station_mask: np.ndarray


@dataclass
class ODData:
    od_node_ids: np.ndarray
    od_node_index: np.ndarray
    non_ev: np.ndarray
    ev: np.ndarray
    ev_node_ids: np.ndarray
    ev_node_index: np.ndarray
    ev_od_positions: np.ndarray
    potential_non_ev: float
    potential_ev: float


@dataclass
class QueueMetrics:
    arrivals_per_hour: np.ndarray
    entries_period: np.ndarray
    offered_load: np.ndarray
    utilization: np.ndarray
    average_wait_hours: np.ndarray
    sojourn_min: np.ndarray
    stable: np.ndarray


@dataclass
class EVPathPool:

    kind: np.ndarray
    station: np.ndarray
    direct_tree: np.ndarray
    first_tree: np.ndarray
    second_tree: np.ndarray
    count: np.ndarray
    distinct_station_count: np.ndarray
    mean_pairwise_jaccard_distance: np.ndarray
    requires_charging: np.ndarray

    direct_variant_pred_edge: Tuple[np.ndarray, ...]
    direct_variant_order: Tuple[np.ndarray, ...]
    direct_variant_miles: np.ndarray
    refresh_iteration: int


@dataclass
class AssignmentResult:
    config: ScenarioConfig
    network: NetworkData
    od: ODData
    total_flow: np.ndarray
    ev_flow: np.ndarray
    non_ev_flow: np.ndarray
    link_time_min: np.ndarray
    station_metrics: QueueMetrics
    served_ev_by_class: np.ndarray
    potential_ev_by_class: np.ndarray
    served_non_ev: float
    iterations: int
    converged: bool
    relative_gap: float
    ev_path_pool: Optional[EVPathPool] = None
    ev_path_pool_summary: Optional[pd.DataFrame] = None
    ev_path_pool_refreshes: int = 0


@njit(cache=False)
def _logaddexp_numba(a: float, b: float) -> float:
    if math.isinf(a) and a < 0:
        return b
    if math.isinf(b) and b < 0:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=False)
def dial_logit_load_all_origins(
    distances: np.ndarray,
    demand: np.ndarray,
    origin_node_index: np.ndarray,
    destination_node_index: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_cost: np.ndarray,
    out_indptr: np.ndarray,
    out_edges: np.ndarray,
    in_indptr: np.ndarray,
    in_edges: np.ndarray,
    theta: float,
) -> Tuple[np.ndarray, float]:

    n_origins, n_nodes = distances.shape
    n_edges = edge_u.shape[0]
    n_dest = destination_node_index.shape[0]
    edge_flow = np.zeros(n_edges, dtype=np.float64)
    unassigned = 0.0
    logw = np.empty(n_nodes, dtype=np.float64)
    node_flow = np.empty(n_nodes, dtype=np.float64)

    for oi in range(n_origins):
        dist = distances[oi]
        order = np.argsort(dist)
        for n in range(n_nodes):
            logw[n] = -np.inf
            node_flow[n] = 0.0
        origin = origin_node_index[oi]
        logw[origin] = 0.0


        for pos in range(n_nodes):
            u = order[pos]
            du = dist[u]
            if not math.isfinite(du) or not math.isfinite(logw[u]):
                continue
            for kk in range(out_indptr[u], out_indptr[u + 1]):
                e = out_edges[kk]
                v = edge_v[e]
                dv = dist[v]
                if not math.isfinite(dv) or not (du + 1.0e-12 < dv):
                    continue
                log_likelihood = theta * (dv - du - edge_cost[e])
                candidate = logw[u] + log_likelihood
                logw[v] = _logaddexp_numba(logw[v], candidate)

        for dj in range(n_dest):
            q = demand[oi, dj]
            if q <= 0.0:
                continue
            node = destination_node_index[dj]
            if math.isfinite(logw[node]):
                node_flow[node] += q
            else:
                unassigned += q


        for pos in range(n_nodes - 1, -1, -1):
            v = order[pos]
            qv = node_flow[v]
            if qv <= 0.0 or not math.isfinite(logw[v]):
                continue
            dv = dist[v]
            for kk in range(in_indptr[v], in_indptr[v + 1]):
                e = in_edges[kk]
                u = edge_u[e]
                du = dist[u]
                if not math.isfinite(du) or not (du + 1.0e-12 < dv):
                    continue
                exponent = logw[u] + theta * (dv - du - edge_cost[e]) - logw[v]
                if exponent < -745.0:
                    continue
                p = math.exp(exponent)
                if p <= 0.0:
                    continue
                f = qv * p
                edge_flow[e] += f
                node_flow[u] += f

    return edge_flow, unassigned


@njit(cache=False)
def cumulative_metric_on_trees(
    predecessor_edge: np.ndarray,
    tree_order: np.ndarray,
    source_node_index: np.ndarray,
    edge_u: np.ndarray,
    edge_metric: np.ndarray,
) -> np.ndarray:

    n_sources, n_nodes = predecessor_edge.shape
    out = np.full((n_sources, n_nodes), np.inf, dtype=np.float64)
    for si in range(n_sources):
        out[si, source_node_index[si]] = 0.0
        order = tree_order[si]
        for pos in range(n_nodes):
            v = order[pos]
            e = predecessor_edge[si, v]
            if e < 0:
                continue
            u = edge_u[e]
            base = out[si, u]
            if math.isfinite(base):
                out[si, v] = base + edge_metric[e]
    return out


@njit(cache=False)
def load_origin_tree_flows(
    predecessor_edge: np.ndarray,
    tree_order: np.ndarray,
    direct_demand: np.ndarray,
    destination_node_index: np.ndarray,
    station_leg_demand: np.ndarray,
    station_entrance_node_index: np.ndarray,
    edge_u: np.ndarray,
    n_edges: int,
) -> np.ndarray:

    n_sources, n_nodes = predecessor_edge.shape
    n_dest = destination_node_index.shape[0]
    n_stations = station_entrance_node_index.shape[0]
    edge_flow = np.zeros(n_edges, dtype=np.float64)
    accum = np.empty(n_nodes, dtype=np.float64)
    for si in range(n_sources):
        for n in range(n_nodes):
            accum[n] = 0.0
        for dj in range(n_dest):
            q = direct_demand[si, dj]
            if q > 0.0:
                accum[destination_node_index[dj]] += q
        for sj in range(n_stations):
            q = station_leg_demand[si, sj]
            if q > 0.0:
                accum[station_entrance_node_index[sj]] += q
        order = tree_order[si]
        for pos in range(n_nodes - 1, -1, -1):
            v = order[pos]
            qv = accum[v]
            if qv <= 0.0:
                continue
            e = predecessor_edge[si, v]
            if e < 0:
                continue
            edge_flow[e] += qv
            accum[edge_u[e]] += qv
    return edge_flow


@njit(cache=False)
def load_direct_tree_flows(
    predecessor_edge: np.ndarray,
    tree_order: np.ndarray,
    direct_demand: np.ndarray,
    destination_node_index: np.ndarray,
    edge_u: np.ndarray,
    n_edges: int,
) -> np.ndarray:

    n_sources, n_nodes = predecessor_edge.shape
    n_dest = destination_node_index.shape[0]
    edge_flow = np.zeros(n_edges, dtype=np.float64)
    accum = np.empty(n_nodes, dtype=np.float64)
    for si in range(n_sources):
        for n in range(n_nodes):
            accum[n] = 0.0
        for dj in range(n_dest):
            q = direct_demand[si, dj]
            if q > 0.0:
                accum[destination_node_index[dj]] += q
        order = tree_order[si]
        for pos in range(n_nodes - 1, -1, -1):
            v = order[pos]
            qv = accum[v]
            if qv <= 0.0:
                continue
            e = predecessor_edge[si, v]
            if e < 0:
                continue
            edge_flow[e] += qv
            accum[edge_u[e]] += qv
    return edge_flow


@njit(cache=False)
def load_station_tree_flows(
    predecessor_edge: np.ndarray,
    tree_order: np.ndarray,
    destination_demand: np.ndarray,
    destination_node_index: np.ndarray,
    edge_u: np.ndarray,
    n_edges: int,
) -> np.ndarray:

    n_sources, n_nodes = predecessor_edge.shape
    n_dest = destination_node_index.shape[0]
    edge_flow = np.zeros(n_edges, dtype=np.float64)
    accum = np.empty(n_nodes, dtype=np.float64)
    for si in range(n_sources):
        for n in range(n_nodes):
            accum[n] = 0.0
        for dj in range(n_dest):
            q = destination_demand[si, dj]
            if q > 0.0:
                accum[destination_node_index[dj]] += q
        order = tree_order[si]
        for pos in range(n_nodes - 1, -1, -1):
            v = order[pos]
            qv = accum[v]
            if qv <= 0.0:
                continue
            e = predecessor_edge[si, v]
            if e < 0:
                continue
            edge_flow[e] += qv
            accum[edge_u[e]] += qv
    return edge_flow


@njit(cache=False)
def path_minhash_to_targets(
    predecessor_edge: np.ndarray,
    source_node_index: np.ndarray,
    target_node_index: np.ndarray,
    edge_u: np.ndarray,
    edge_minhash: np.ndarray,
) -> np.ndarray:

    n_sources = predecessor_edge.shape[0]
    n_targets = target_node_index.shape[0]
    n_nodes = predecessor_edge.shape[1]
    n_hash = edge_minhash.shape[1]
    max_u64 = np.uint64(18446744073709551615)
    out = np.full((n_sources, n_targets, n_hash), max_u64, dtype=np.uint64)

    for si in range(n_sources):
        source = source_node_index[si]
        for tj in range(n_targets):
            current = target_node_index[tj]
            steps = 0
            while current != source and steps <= n_nodes:
                e = predecessor_edge[si, current]
                if e < 0:
                    break
                for hh in range(n_hash):
                    value = edge_minhash[e, hh]
                    if value < out[si, tj, hh]:
                        out[si, tj, hh] = value
                current = edge_u[e]
                steps += 1
    return out


def build_edge_minhash(
    edge_type: np.ndarray,
    n_hashes: int,
    seed: int,
) -> np.ndarray:

    if n_hashes <= 0:
        raise ValueError("ev_path_minhash_size must be positive.")
    n_edges = len(edge_type)
    edge_index = np.arange(n_edges, dtype=np.uint64)
    hashes = np.empty((n_edges, n_hashes), dtype=np.uint64)
    mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        for hh in range(n_hashes):
            x = edge_index + np.uint64(seed) + np.uint64(hh + 1) * np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = (x ^ (x >> np.uint64(31))) & mask64
            hashes[:, hh] = x
    hashes[np.asarray(edge_type) != "road", :] = np.uint64(0xFFFFFFFFFFFFFFFF)
    return hashes


@njit(cache=False)
def _signature_equal(a: np.ndarray, b: np.ndarray) -> bool:
    for hh in range(a.shape[0]):
        if a[hh] != b[hh]:
            return False
    return True


@njit(cache=False)
def _signature_jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    matches = 0
    n_hash = a.shape[0]
    for hh in range(n_hash):
        if a[hh] == b[hh]:
            matches += 1
    return 1.0 - float(matches) / float(max(1, n_hash))


@njit(cache=False)
def build_diverse_ev_path_pool(
    ev_demand: np.ndarray,
    class_shares: np.ndarray,
    initial_ranges: np.ndarray,
    full_range: float,
    direct_variant_cost: np.ndarray,
    direct_variant_miles: np.ndarray,
    direct_shortest_miles: np.ndarray,
    first_time_cost: np.ndarray,
    first_time_miles: np.ndarray,
    first_distance_cost: np.ndarray,
    first_shortest_miles: np.ndarray,
    second_time_cost: np.ndarray,
    second_time_miles: np.ndarray,
    second_distance_cost: np.ndarray,
    second_shortest_miles: np.ndarray,
    station_sojourn_min: np.ndarray,
    direct_variant_sig: np.ndarray,
    first_time_sig: np.ndarray,
    first_distance_sig: np.ndarray,
    second_time_sig: np.ndarray,
    second_distance_sig: np.ndarray,
    k_paths: int,
    prefer_distinct_stations: bool,
    diversity_weight: float,
    cost_weight: float,
    unstable_loading_penalty_min: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    n_orig, n_dest = ev_demand.shape
    n_classes = class_shares.shape[0]
    n_stations = station_sojourn_min.shape[0]
    n_direct_variants = direct_variant_cost.shape[0]
    n_hash = direct_variant_sig.shape[3]
    max_alt = n_direct_variants + 4 * n_stations

    pool_kind = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_station = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int16)
    pool_direct_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_first_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_second_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_count = np.zeros((n_classes, n_orig, n_dest), dtype=np.int8)
    pool_distinct_stations = np.zeros((n_classes, n_orig, n_dest), dtype=np.int8)
    pool_mean_jaccard_distance = np.zeros((n_classes, n_orig, n_dest), dtype=np.float32)
    pool_requires_charging = np.zeros((n_classes, n_orig, n_dest), dtype=np.uint8)

    cand_cost = np.empty(max_alt, dtype=np.float64)
    cand_kind = np.empty(max_alt, dtype=np.int8)
    cand_station = np.empty(max_alt, dtype=np.int16)
    cand_direct_tree = np.empty(max_alt, dtype=np.int8)
    cand_first_tree = np.empty(max_alt, dtype=np.int8)
    cand_second_tree = np.empty(max_alt, dtype=np.int8)
    cand_sig = np.empty((max_alt, n_hash), dtype=np.uint64)
    selected = np.zeros(max_alt, dtype=np.uint8)
    selected_idx = np.empty(k_paths, dtype=np.int16)
    used_station = np.zeros(max(1, n_stations), dtype=np.uint8)

    for ci in range(n_classes):
        share = class_shares[ci]
        r0 = initial_ranges[ci]
        if share <= 0.0:
            continue
        for oi in range(n_orig):
            for dj in range(n_dest):
                if ev_demand[oi, dj] <= 0.0:
                    continue


                direct_possible = (
                    direct_shortest_miles[oi, dj] <= r0 + 1.0e-9
                    and math.isfinite(direct_shortest_miles[oi, dj])
                )
                pool_requires_charging[ci, oi, dj] = 0 if direct_possible else 1
                n_alt = 0

                if direct_possible:


                    for vv in range(n_direct_variants):
                        route_cost = direct_variant_cost[vv, oi, dj]
                        route_miles = direct_variant_miles[vv, oi, dj]
                        feasible = (
                            route_miles <= r0 + 1.0e-9
                            and math.isfinite(route_cost)
                        )
                        if not feasible:
                            continue

                        duplicate = -1
                        for aa in range(n_alt):
                            same = True
                            for hh in range(n_hash):
                                if cand_sig[aa, hh] != direct_variant_sig[vv, oi, dj, hh]:
                                    same = False
                                    break
                            if same:
                                duplicate = aa
                                break
                        if duplicate >= 0:
                            if route_cost < cand_cost[duplicate]:
                                cand_cost[duplicate] = route_cost
                                cand_direct_tree[duplicate] = vv
                            continue

                        cand_cost[n_alt] = route_cost
                        cand_kind[n_alt] = 0
                        cand_station[n_alt] = -1
                        cand_direct_tree[n_alt] = vv
                        cand_first_tree[n_alt] = -1
                        cand_second_tree[n_alt] = -1
                        for hh in range(n_hash):
                            cand_sig[n_alt, hh] = direct_variant_sig[vv, oi, dj, hh]
                        n_alt += 1

                else:


                    for sj in range(n_stations):
                        for first_mode in range(2):
                            if first_mode == 0:
                                first_ok = (
                                    first_time_miles[oi, sj] <= r0 + 1.0e-9
                                    and math.isfinite(first_time_cost[oi, sj])
                                )
                                first_cost = first_time_cost[oi, sj]
                            else:
                                first_ok = (
                                    first_shortest_miles[oi, sj] <= r0 + 1.0e-9
                                    and math.isfinite(first_distance_cost[oi, sj])
                                )
                                first_cost = first_distance_cost[oi, sj]
                            if not first_ok:
                                continue

                            for second_mode in range(2):
                                if second_mode == 0:
                                    second_ok = (
                                        second_time_miles[sj, dj] <= full_range + 1.0e-9
                                        and math.isfinite(second_time_cost[sj, dj])
                                    )
                                    second_cost = second_time_cost[sj, dj]
                                else:
                                    second_ok = (
                                        second_shortest_miles[sj, dj] <= full_range + 1.0e-9
                                        and math.isfinite(second_distance_cost[sj, dj])
                                    )
                                    second_cost = second_distance_cost[sj, dj]
                                if not second_ok:
                                    continue

                                road_cost = first_cost + second_cost
                                sojourn = station_sojourn_min[sj]
                                route_cost = road_cost + (
                                    sojourn if math.isfinite(sojourn)
                                    else unstable_loading_penalty_min
                                )

                                duplicate = -1
                                for aa in range(n_alt):
                                    if cand_station[aa] != sj:
                                        continue
                                    same = True
                                    for hh in range(n_hash):
                                        first_sig_value = (
                                            first_time_sig[oi, sj, hh]
                                            if first_mode == 0 else first_distance_sig[oi, sj, hh]
                                        )
                                        second_sig_value = (
                                            second_time_sig[sj, dj, hh]
                                            if second_mode == 0 else second_distance_sig[sj, dj, hh]
                                        )
                                        sig_value = (
                                            first_sig_value
                                            if first_sig_value < second_sig_value
                                            else second_sig_value
                                        )
                                        if cand_sig[aa, hh] != sig_value:
                                            same = False
                                            break
                                    if same:
                                        duplicate = aa
                                        break
                                if duplicate >= 0:
                                    if route_cost < cand_cost[duplicate]:
                                        cand_cost[duplicate] = route_cost
                                        cand_first_tree[duplicate] = first_mode
                                        cand_second_tree[duplicate] = second_mode
                                    continue

                                cand_cost[n_alt] = route_cost
                                cand_kind[n_alt] = 1
                                cand_station[n_alt] = sj
                                cand_direct_tree[n_alt] = -1
                                cand_first_tree[n_alt] = first_mode
                                cand_second_tree[n_alt] = second_mode
                                for hh in range(n_hash):
                                    first_sig_value = (
                                        first_time_sig[oi, sj, hh]
                                        if first_mode == 0 else first_distance_sig[oi, sj, hh]
                                    )
                                    second_sig_value = (
                                        second_time_sig[sj, dj, hh]
                                        if second_mode == 0 else second_distance_sig[sj, dj, hh]
                                    )
                                    cand_sig[n_alt, hh] = (
                                        first_sig_value
                                        if first_sig_value < second_sig_value
                                        else second_sig_value
                                    )
                                n_alt += 1

                if n_alt <= 0:
                    continue

                for aa in range(n_alt):
                    selected[aa] = 0
                for sj in range(max(1, n_stations)):
                    used_station[sj] = 0
                selected_count = 0
                cheapest_cost = np.inf
                for aa in range(n_alt):
                    if cand_cost[aa] < cheapest_cost:
                        cheapest_cost = cand_cost[aa]

                while selected_count < k_paths and selected_count < n_alt:
                    best_a = -1
                    best_score = -1.0e300
                    best_cost = np.inf

                    novel_station_exists = False
                    if (
                        not direct_possible
                        and prefer_distinct_stations
                        and selected_count > 0
                    ):
                        for aa in range(n_alt):
                            if selected[aa] == 1:
                                continue
                            sj = cand_station[aa]
                            if sj >= 0 and used_station[sj] == 0:
                                novel_station_exists = True
                                break

                    for aa in range(n_alt):
                        if selected[aa] == 1:
                            continue
                        if novel_station_exists:
                            sj = cand_station[aa]
                            if sj < 0 or used_station[sj] == 1:
                                continue

                        if selected_count == 0:
                            score = -cand_cost[aa]
                        else:
                            min_distance = 1.0
                            for bb in range(selected_count):
                                prev = selected_idx[bb]
                                matches = 0
                                for hh in range(n_hash):
                                    if cand_sig[aa, hh] == cand_sig[prev, hh]:
                                        matches += 1
                                distance = 1.0 - float(matches) / float(max(1, n_hash))
                                if distance < min_distance:
                                    min_distance = distance
                            relative_cost = (
                                (cand_cost[aa] - cheapest_cost)
                                / max(1.0, abs(cheapest_cost))
                            )
                            score = (
                                diversity_weight * min_distance
                                - cost_weight * relative_cost
                            )

                        if score > best_score + 1.0e-12 or (
                            abs(score - best_score) <= 1.0e-12
                            and cand_cost[aa] < best_cost
                        ):
                            best_score = score
                            best_cost = cand_cost[aa]
                            best_a = aa

                    if best_a < 0:
                        break
                    selected[best_a] = 1
                    selected_idx[selected_count] = best_a
                    pool_kind[ci, oi, dj, selected_count] = cand_kind[best_a]
                    pool_station[ci, oi, dj, selected_count] = cand_station[best_a]
                    pool_direct_tree[ci, oi, dj, selected_count] = cand_direct_tree[best_a]
                    pool_first_tree[ci, oi, dj, selected_count] = cand_first_tree[best_a]
                    pool_second_tree[ci, oi, dj, selected_count] = cand_second_tree[best_a]
                    if cand_kind[best_a] == 1:
                        sj = cand_station[best_a]
                        if sj >= 0:
                            used_station[sj] = 1
                    selected_count += 1

                pool_count[ci, oi, dj] = selected_count
                distinct = 0
                for sj in range(n_stations):
                    if used_station[sj] == 1:
                        distinct += 1
                pool_distinct_stations[ci, oi, dj] = distinct

                if selected_count >= 2:
                    total_distance = 0.0
                    pair_count = 0
                    for aa in range(selected_count):
                        for bb in range(aa + 1, selected_count):
                            ia = selected_idx[aa]
                            ib = selected_idx[bb]
                            matches = 0
                            for hh in range(n_hash):
                                if cand_sig[ia, hh] == cand_sig[ib, hh]:
                                    matches += 1
                            total_distance += 1.0 - float(matches) / float(max(1, n_hash))
                            pair_count += 1
                    pool_mean_jaccard_distance[ci, oi, dj] = (
                        total_distance / float(pair_count)
                    )

    return (
        pool_kind, pool_station, pool_direct_tree, pool_first_tree,
        pool_second_tree, pool_count, pool_distinct_stations,
        pool_mean_jaccard_distance, pool_requires_charging,
    )


@njit(cache=False)
def ev_k_path_logit_loading(
    ev_demand: np.ndarray,
    class_shares: np.ndarray,
    initial_ranges: np.ndarray,
    full_range: float,
    theta: float,
    pool_kind: np.ndarray,
    pool_station: np.ndarray,
    pool_direct_tree: np.ndarray,
    pool_first_tree: np.ndarray,
    pool_second_tree: np.ndarray,
    pool_count: np.ndarray,
    direct_variant_cost: np.ndarray,
    direct_variant_miles: np.ndarray,
    first_time_cost: np.ndarray,
    first_time_miles: np.ndarray,
    first_distance_cost: np.ndarray,
    first_shortest_miles: np.ndarray,
    second_time_cost: np.ndarray,
    second_time_miles: np.ndarray,
    second_distance_cost: np.ndarray,
    second_shortest_miles: np.ndarray,
    station_sojourn_min: np.ndarray,
    unstable_loading_penalty_min: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    n_orig, n_dest = ev_demand.shape
    n_classes = class_shares.shape[0]
    n_stations = station_sojourn_min.shape[0]
    n_direct_variants = direct_variant_cost.shape[0]
    k_paths = pool_kind.shape[3]

    direct_variant_q = np.zeros(
        (n_direct_variants, n_orig, n_dest), dtype=np.float64
    )
    first_time_q = np.zeros((n_orig, n_stations), dtype=np.float64)
    first_dist_q = np.zeros((n_orig, n_stations), dtype=np.float64)
    second_time_q = np.zeros((n_stations, n_dest), dtype=np.float64)
    second_dist_q = np.zeros((n_stations, n_dest), dtype=np.float64)
    station_entries = np.zeros(n_stations, dtype=np.float64)
    served_by_class = np.zeros(n_classes, dtype=np.float64)
    potential_by_class = np.zeros(n_classes, dtype=np.float64)
    unserved_by_class = np.zeros(n_classes, dtype=np.float64)

    costs = np.empty(k_paths, dtype=np.float64)
    road_costs = np.empty(k_paths, dtype=np.float64)
    valid_slot = np.zeros(k_paths, dtype=np.uint8)
    actual_direct_variant = np.full(k_paths, -1, dtype=np.int8)
    actual_first_mode = np.full(k_paths, -1, dtype=np.int8)
    actual_second_mode = np.full(k_paths, -1, dtype=np.int8)
    weights = np.empty(k_paths, dtype=np.float64)

    for ci in range(n_classes):
        share = class_shares[ci]
        r0 = initial_ranges[ci]
        for oi in range(n_orig):
            for dj in range(n_dest):
                q = ev_demand[oi, dj] * share
                if q <= 0.0:
                    continue
                potential_by_class[ci] += q
                selected_count = int(pool_count[ci, oi, dj])
                n_valid = 0
                for kk in range(k_paths):
                    valid_slot[kk] = 0
                    actual_direct_variant[kk] = -1
                    actual_first_mode[kk] = -1
                    actual_second_mode[kk] = -1
                    costs[kk] = np.inf
                    road_costs[kk] = np.inf

                for kk in range(selected_count):
                    kind = int(pool_kind[ci, oi, dj, kk])
                    if kind == 0:
                        vv = int(pool_direct_tree[ci, oi, dj, kk])
                        if vv < 0 or vv >= n_direct_variants:
                            continue
                        road = direct_variant_cost[vv, oi, dj]
                        miles = direct_variant_miles[vv, oi, dj]
                        if miles <= r0 + 1.0e-9 and math.isfinite(road):
                            valid_slot[kk] = 1
                            actual_direct_variant[kk] = vv
                            road_costs[kk] = road
                            costs[kk] = road
                            n_valid += 1

                    elif kind == 1:
                        sj = int(pool_station[ci, oi, dj, kk])
                        if sj < 0 or sj >= n_stations:
                            continue

                        preferred_first = int(pool_first_tree[ci, oi, dj, kk])
                        first_time_ok = (
                            first_time_miles[oi, sj] <= r0 + 1.0e-9
                            and math.isfinite(first_time_cost[oi, sj])
                        )
                        first_distance_ok = (
                            first_shortest_miles[oi, sj] <= r0 + 1.0e-9
                            and math.isfinite(first_distance_cost[oi, sj])
                        )
                        first_mode = -1
                        first = np.inf
                        if preferred_first == 0 and first_time_ok:
                            first_mode = 0
                            first = first_time_cost[oi, sj]
                        elif preferred_first == 1 and first_distance_ok:
                            first_mode = 1
                            first = first_distance_cost[oi, sj]
                        elif first_distance_ok:
                            first_mode = 1
                            first = first_distance_cost[oi, sj]
                        elif first_time_ok:
                            first_mode = 0
                            first = first_time_cost[oi, sj]

                        preferred_second = int(pool_second_tree[ci, oi, dj, kk])
                        second_time_ok = (
                            second_time_miles[sj, dj] <= full_range + 1.0e-9
                            and math.isfinite(second_time_cost[sj, dj])
                        )
                        second_distance_ok = (
                            second_shortest_miles[sj, dj] <= full_range + 1.0e-9
                            and math.isfinite(second_distance_cost[sj, dj])
                        )
                        second_mode = -1
                        second = np.inf
                        if preferred_second == 0 and second_time_ok:
                            second_mode = 0
                            second = second_time_cost[sj, dj]
                        elif preferred_second == 1 and second_distance_ok:
                            second_mode = 1
                            second = second_distance_cost[sj, dj]
                        elif second_distance_ok:
                            second_mode = 1
                            second = second_distance_cost[sj, dj]
                        elif second_time_ok:
                            second_mode = 0
                            second = second_time_cost[sj, dj]

                        if first_mode >= 0 and second_mode >= 0:
                            road = first + second
                            sojourn = station_sojourn_min[sj]
                            valid_slot[kk] = 1
                            actual_first_mode[kk] = first_mode
                            actual_second_mode[kk] = second_mode
                            road_costs[kk] = road
                            costs[kk] = (
                                road + sojourn if math.isfinite(sojourn)
                                else np.inf
                            )
                            n_valid += 1

                if n_valid == 0:
                    unserved_by_class[ci] += q
                    continue
                served_by_class[ci] += q

                finite_count = 0
                cmin = np.inf
                for kk in range(selected_count):
                    if valid_slot[kk] == 1 and math.isfinite(costs[kk]):
                        finite_count += 1
                        if costs[kk] < cmin:
                            cmin = costs[kk]

                denom = 0.0
                if finite_count > 0:
                    for kk in range(selected_count):
                        if valid_slot[kk] == 1 and math.isfinite(costs[kk]):
                            exponent = -theta * (costs[kk] - cmin)
                            weights[kk] = (
                                0.0 if exponent < -745.0 else math.exp(exponent)
                            )
                            denom += weights[kk]
                        else:
                            weights[kk] = 0.0
                else:
                    cmin = np.inf
                    for kk in range(selected_count):
                        if valid_slot[kk] == 1:
                            costs[kk] = (
                                road_costs[kk] + unstable_loading_penalty_min
                            )
                            if costs[kk] < cmin:
                                cmin = costs[kk]
                    for kk in range(selected_count):
                        if valid_slot[kk] == 1:
                            exponent = -theta * (costs[kk] - cmin)
                            weights[kk] = (
                                0.0 if exponent < -745.0 else math.exp(exponent)
                            )
                            denom += weights[kk]
                        else:
                            weights[kk] = 0.0

                if denom <= 0.0 or not math.isfinite(denom):
                    best_k = -1
                    best_cost = np.inf
                    for kk in range(selected_count):
                        if valid_slot[kk] == 1 and costs[kk] < best_cost:
                            best_cost = costs[kk]
                            best_k = kk
                    if best_k < 0:
                        unserved_by_class[ci] += q
                        served_by_class[ci] -= q
                        continue
                    for kk in range(selected_count):
                        weights[kk] = 1.0 if kk == best_k else 0.0
                    denom = 1.0

                for kk in range(selected_count):
                    if weights[kk] <= 0.0:
                        continue
                    f = q * weights[kk] / denom
                    kind = int(pool_kind[ci, oi, dj, kk])
                    if kind == 0:
                        vv = int(actual_direct_variant[kk])
                        direct_variant_q[vv, oi, dj] += f
                    else:
                        sj = int(pool_station[ci, oi, dj, kk])
                        station_entries[sj] += f
                        if int(actual_first_mode[kk]) == 0:
                            first_time_q[oi, sj] += f
                        else:
                            first_dist_q[oi, sj] += f
                        if int(actual_second_mode[kk]) == 0:
                            second_time_q[sj, dj] += f
                        else:
                            second_dist_q[sj, dj] += f

    return (
        direct_variant_q,
        first_time_q, first_dist_q,
        second_time_q, second_dist_q,
        station_entries, served_by_class,
        potential_by_class, unserved_by_class,
    )


def summarize_ev_path_pool(
    pool: EVPathPool,
    ev_demand: np.ndarray,
    class_shares: np.ndarray,
    soc_fractions: Sequence[float],
    battery_range_miles: float,
    k_target: int,
) -> pd.DataFrame:

    positive = ev_demand > 0.0
    rows: List[Dict[str, float]] = []
    for ci, share in enumerate(class_shares):
        weights = ev_demand * float(share)
        feasible = positive & (pool.count[ci] > 0)
        direct_mask = positive & (pool.requires_charging[ci] == 0)
        charging_mask = positive & (pool.requires_charging[ci] == 1)
        total_weight = float(weights[positive].sum())
        feasible_weight = float(weights[feasible].sum())
        direct_weight = float(weights[direct_mask].sum())
        charging_weight = float(weights[charging_mask].sum())


        direct_station_violation = np.zeros_like(direct_mask)
        for kk in range(pool.kind.shape[3]):
            direct_station_violation |= direct_mask & (pool.kind[ci, :, :, kk] == 1)
        violation_weight = float(weights[direct_station_violation].sum())

        if feasible_weight > 0.0:
            avg_paths = float(
                np.sum(weights[feasible] * pool.count[ci][feasible])
                / feasible_weight
            )


            charging_feasible_for_station_metric = (
                feasible & (pool.requires_charging[ci] == 1)
            )
            charging_metric_weight = float(
                weights[charging_feasible_for_station_metric].sum()
            )
            avg_stations = (
                float(
                    np.sum(
                        weights[charging_feasible_for_station_metric]
                        * pool.distinct_station_count[ci][
                            charging_feasible_for_station_metric
                        ]
                    )
                    / charging_metric_weight
                )
                if charging_metric_weight > 0.0 else 0.0
            )
            avg_diversity = float(
                np.sum(
                    weights[feasible]
                    * pool.mean_pairwise_jaccard_distance[ci][feasible]
                )
                / feasible_weight
            )
            share_full_k = float(
                weights[feasible & (pool.count[ci] >= k_target)].sum()
                / feasible_weight
            )
            share_two_stations = (
                float(
                    weights[
                        charging_feasible_for_station_metric
                        & (pool.distinct_station_count[ci] >= 2)
                    ].sum()
                    / charging_metric_weight
                )
                if charging_metric_weight > 0.0 else 0.0
            )
            share_three_stations = (
                float(
                    weights[
                        charging_feasible_for_station_metric
                        & (pool.distinct_station_count[ci] >= 3)
                    ].sum()
                    / charging_metric_weight
                )
                if charging_metric_weight > 0.0 else 0.0
            )
        else:
            avg_paths = avg_stations = avg_diversity = share_full_k = float("nan")
            share_two_stations = share_three_stations = float("nan")

        direct_feasible = direct_mask & (pool.count[ci] > 0)
        charging_feasible = charging_mask & (pool.count[ci] > 0)
        direct_feasible_weight = float(weights[direct_feasible].sum())
        charging_feasible_weight = float(weights[charging_feasible].sum())
        avg_direct_routes = (
            float(
                np.sum(weights[direct_feasible] * pool.count[ci][direct_feasible])
                / direct_feasible_weight
            )
            if direct_feasible_weight > 0.0 else float("nan")
        )
        avg_charging_routes = (
            float(
                np.sum(
                    weights[charging_feasible]
                    * pool.count[ci][charging_feasible]
                )
                / charging_feasible_weight
            )
            if charging_feasible_weight > 0.0 else float("nan")
        )

        rows.append({
            "soc_fraction": float(soc_fractions[ci]),
            "initial_range_miles": float(soc_fractions[ci]) * float(battery_range_miles),
            "potential_ev_demand": total_weight,
            "direct_trip_possible_demand": direct_weight,
            "charging_required_demand": charging_weight,
            "energy_feasible_ev_demand_in_pool": feasible_weight,
            "average_number_of_routes": avg_paths,
            "average_direct_only_routes_when_no_charge_needed": avg_direct_routes,
            "average_charging_routes_when_charge_required": avg_charging_routes,
            "target_k_routes": float(k_target),
            "share_feasible_demand_with_full_k": share_full_k,
            "average_distinct_charging_stations": avg_stations,
            "share_feasible_demand_with_at_least_2_station_choices": share_two_stations,
            "share_feasible_demand_with_at_least_3_station_choices": share_three_stations,
            "direct_trip_pool_charging_route_violation_demand": violation_weight,
            "mean_pairwise_road_jaccard_distance_minhash": avg_diversity,
            "pool_refresh_iteration": float(pool.refresh_iteration),
        })
    return pd.DataFrame(rows)


STATION_IDS = tuple(f"C{i}" for i in range(1, 10))
ROAD_SOURCE_EDGE_TYPES = {
    "road_original",
    "road_split_upstream",
    "road_split_downstream",
}
SOURCE_TO_MODEL_EDGE_TYPE = {
    "road_original": "road",
    "road_split_upstream": "road",
    "road_split_downstream": "road",
    "station_entry": "in",
    "station_exit": "out",
    "station_bypass": "byp",
}
CLOSURE_DEPTH_MM = 300.0
DCFC_DWELL_MIN = 42.0
LEVEL2_DWELL_MIN = 120.0


def _normalize_node_label(value: object) -> str:

    label = str(value).strip()
    if not label:
        raise ValueError("Encountered an empty node label.")
    if re.fullmatch(r"[+-]?\d+\.0+", label):
        label = str(int(float(label)))
    return label


def _parse_bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_bool_series(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    ).to_numpy(dtype=np.bool_)


def locate_local_csv(root: Path, canonical_name: str) -> Path:

    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    direct = root / canonical_name
    if direct.is_file():
        return direct

    p = Path(canonical_name)
    pattern = re.compile(
        rf"^{re.escape(p.stem)}(?:\((\d+)\))?{re.escape(p.suffix)}$",
        flags=re.IGNORECASE,
    )

    matches: List[Tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.match(candidate.name)
        if match:
            number = int(match.group(1)) if match.group(1) is not None else 0
            matches.append((number, candidate))


    if not matches:
        for candidate in root.rglob(f"{p.stem}*{p.suffix}"):
            if not candidate.is_file():
                continue
            match = pattern.match(candidate.name)
            if match:
                number = int(match.group(1)) if match.group(1) is not None else 0
                matches.append((number, candidate))

    if not matches:
        nearby = sorted(x.name for x in root.glob("*.csv"))
        preview = "\n  ".join(nearby[:30]) if nearby else "(no CSV files found)"
        raise FileNotFoundError(
            f"Could not find raw local file {canonical_name!r} under:\n"
            f"  {root}\n"
            f"CSV files visible in that folder include:\n  {preview}"
        )

    matches.sort(key=lambda item: (item[0], item[1].name.lower()))
    selected = matches[-1][1]
    if len(matches) > 1:
        print(
            f"Multiple copies match {canonical_name}; using {selected.name}",
            flush=True,
        )
    return selected


def read_od_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    row_ids = df.index.astype(int).to_numpy(dtype=np.int64)
    col_ids = np.asarray([int(c) for c in df.columns], dtype=np.int64)
    if not np.array_equal(row_ids, col_ids):
        raise ValueError(f"OD row and column orders differ in {path}")
    matrix = df.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1.0e-12):
        raise ValueError(f"Invalid OD demand values in {path}")
    matrix[matrix < 0.0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return row_ids, matrix


def retained_capacity_ratio(depth_mm: np.ndarray) -> np.ndarray:

    depth = np.maximum(np.asarray(depth_mm, dtype=np.float64), 0.0)
    return np.select(
        [
            depth < 5.0,
            depth < 10.0,
            depth < 50.0,
            depth < 100.0,
            depth < 150.0,
            depth < 200.0,
            depth < 300.0,
        ],
        [1.0, 0.767, 0.742, 0.637, 0.564, 0.509, 0.463],
        default=0.0,
    ).astype(np.float64)


def flood_safe_speed_kmh(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.maximum(np.asarray(depth_mm, dtype=np.float64), 0.0)
    safe = 0.0009 * depth * depth - 0.5529 * depth + 86.9448
    safe = np.maximum(safe, 0.0)
    safe[depth >= CLOSURE_DEPTH_MM] = 0.0
    return safe


def build_numeric_node_mapping(
    node_labels: Sequence[object],
    extra_labels: Sequence[object],
) -> Dict[str, int]:

    ordered: List[str] = []
    seen: set[str] = set()
    for value in list(node_labels) + list(extra_labels):
        label = _normalize_node_label(value)
        if label not in seen:
            seen.add(label)
            ordered.append(label)

    numeric_ids = [int(label) for label in ordered if re.fullmatch(r"[+-]?\d+", label)]
    if not numeric_ids:
        raise ValueError("No numeric road-network node IDs were found.")
    next_id = max(numeric_ids) + 1

    mapping: Dict[str, int] = {}
    for label in ordered:
        if re.fullmatch(r"[+-]?\d+", label):
            mapping[label] = int(label)
        else:
            mapping[label] = next_id
            next_id += 1

    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("The internal numeric node mapping is not one-to-one.")
    return mapping


def infer_lanes_from_source_capacity(
    source_capacity_vehph: np.ndarray,
    is_road: np.ndarray,
) -> np.ndarray:

    capacity = np.asarray(source_capacity_vehph, dtype=np.float64)
    road_mask = np.asarray(is_road, dtype=np.bool_)
    lanes = np.zeros(capacity.shape, dtype=np.int16)

    valid = road_mask & np.isfinite(capacity) & (capacity > 0.0)


    inferred = np.ceil(capacity[valid] / CAPACITY_PER_LANE_VEHPH - 1.0e-12)
    inferred = np.clip(inferred, 1.0, float(MAX_INFERRED_LANES))
    lanes[valid] = inferred.astype(np.int16)

    invalid_road = road_mask & ~valid
    if np.any(invalid_road):
        bad = np.where(invalid_road)[0][:10].tolist()
        raise ValueError(
            "Physical road links must have a finite positive source capacity; "
            f"first invalid row positions: {bad}"
        )
    return lanes


def _station_service_time_min(level2_ports: int, dcfc_ports: int) -> float:
    total = int(level2_ports) + int(dcfc_ports)
    if total <= 0:
        raise ValueError("Each charging cluster must contain at least one port.")
    return (
        LEVEL2_DWELL_MIN * int(level2_ports)
        + DCFC_DWELL_MIN * int(dcfc_ports)
    ) / total


def prepare_network_from_raw_files(
    net_path: Path,
    topology_node_path: Path,
    station_source_path: Path,
    config: ScenarioConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    raw_net = pd.read_csv(net_path, dtype=str, keep_default_na=False)
    raw_nodes = pd.read_csv(topology_node_path, dtype=str, keep_default_na=False)
    station_source = pd.read_csv(
        station_source_path, dtype=str, keep_default_na=False
    )

    required_net = {
        "init_node", "term_node", "capacity", "length", "free_flow_time",
        "speed", "edge_type", "station_group", "edge_depth_mm",
        "flood_affected", "retain_in_flood_network",
    }
    missing_net = sorted(required_net - set(raw_net.columns))
    if missing_net:
        raise ValueError(f"Raw network file {net_path.name} is missing {missing_net}")

    required_nodes = {"node", "level2", "dcfc", "node_type", "charging_group"}
    missing_nodes = sorted(required_nodes - set(station_source.columns))
    if missing_nodes:
        raise ValueError(
            f"Raw station-source node file {station_source_path.name} is missing {missing_nodes}"
        )
    if "node" not in raw_nodes.columns:
        raise ValueError(f"Raw topology node file {topology_node_path.name} lacks 'node'.")

    raw_net = raw_net.copy()
    raw_nodes = raw_nodes.copy()
    station_source = station_source.copy()
    raw_net["init_node"] = raw_net["init_node"].map(_normalize_node_label)
    raw_net["term_node"] = raw_net["term_node"].map(_normalize_node_label)
    raw_nodes["node"] = raw_nodes["node"].map(_normalize_node_label)
    station_source["node"] = station_source["node"].map(_normalize_node_label)

    node_map = build_numeric_node_mapping(
        raw_nodes["node"].tolist(),
        raw_net["init_node"].tolist() + raw_net["term_node"].tolist(),
    )

    source_edge_type = raw_net["edge_type"].astype(str).str.strip().str.lower()
    unsupported = sorted(set(source_edge_type) - set(SOURCE_TO_MODEL_EDGE_TYPE))
    if unsupported:
        raise ValueError(f"Unsupported raw edge_type values: {unsupported}")
    model_type = source_edge_type.map(SOURCE_TO_MODEL_EDGE_TYPE).to_numpy(dtype=str)
    is_road = model_type == "road"

    def numeric_column(name: str, default: float = 0.0) -> np.ndarray:
        values = pd.to_numeric(raw_net[name], errors="coerce").fillna(default)
        return values.to_numpy(dtype=np.float64)

    length_miles = np.where(is_road, numeric_column("length"), 0.0)


    source_capacity_raw = np.where(is_road, numeric_column("capacity"), 0.0)
    source_free_flow_time_file = np.where(
        is_road, numeric_column("free_flow_time"), 0.0
    )
    source_speed_mph = np.where(is_road, numeric_column("speed"), 0.0)

    inferred_lanes = infer_lanes_from_source_capacity(
        source_capacity_raw, is_road
    )
    capacity_dry = np.where(
        is_road,
        inferred_lanes.astype(np.float64) * CAPACITY_PER_LANE_VEHPH,
        0.0,
    )


    t0_dry = np.zeros(len(raw_net), dtype=np.float64)
    valid_time = (
        is_road
        & np.isfinite(length_miles)
        & (length_miles >= 0.0)
        & np.isfinite(source_speed_mph)
        & (source_speed_mph > 0.0)
    )
    t0_dry[valid_time] = (
        60.0 * length_miles[valid_time] / source_speed_mph[valid_time]
    )
    invalid_time_road = is_road & ~valid_time
    if np.any(invalid_time_road):
        bad = np.where(invalid_time_road)[0][:10].tolist()
        raise ValueError(
            "Physical road links must have finite nonnegative length and "
            f"positive speed; first invalid row positions: {bad}"
        )

    speed_limit_kmh = np.where(
        is_road, source_speed_mph * MILES_TO_KM, 0.0
    )
    depth_mm = np.where(is_road, numeric_column("edge_depth_mm"), 0.0)
    flood_affected = _parse_bool_series(raw_net["flood_affected"]) & is_road
    retained_flag = _parse_bool_series(raw_net["retain_in_flood_network"])

    capacity_ratio = np.ones(len(raw_net), dtype=np.float64)
    capacity_ratio[flood_affected] = retained_capacity_ratio(depth_mm[flood_affected])
    capacity_flood = capacity_dry * capacity_ratio

    safe_speed = speed_limit_kmh.copy()
    safe_speed[flood_affected] = flood_safe_speed_kmh(depth_mm[flood_affected])
    speed_after_flood = np.where(
        speed_limit_kmh > 0.0,
        np.minimum(speed_limit_kmh, safe_speed),
        safe_speed,
    )

    t0_flood = t0_dry.copy()
    bad_flood = (
        is_road
        & flood_affected
        & ((speed_after_flood <= 0.0) | (capacity_flood <= 0.0))
    )
    scalable = (
        is_road
        & flood_affected
        & ~bad_flood
        & (t0_dry > 0.0)
        & (speed_limit_kmh > 0.0)
    )


    t0_flood[scalable] = (
        60.0
        * length_miles[scalable]
        * MILES_TO_KM
        / speed_after_flood[scalable]
    )
    t0_flood[bad_flood] = np.inf


    is_closed = (
        is_road
        & (
            (depth_mm >= CLOSURE_DEPTH_MM)
            | (capacity_flood <= 0.0)
            | ~np.isfinite(t0_flood)
            | (~retained_flag)
        )
    )
    capacity_flood[is_closed] = 0.0
    t0_flood[is_closed] = np.inf

    link_width = max(6, len(str(len(raw_net))))
    link_ids = np.asarray(
        [f"CRL{i:0{link_width}d}" for i in range(1, len(raw_net) + 1)],
        dtype=object,
    )
    station_id = (
        raw_net["station_group"].astype(str).str.strip().str.upper().to_numpy(dtype=str)
    )
    from_label = raw_net["init_node"].to_numpy(dtype=str)
    to_label = raw_net["term_node"].to_numpy(dtype=str)
    from_node = np.asarray([node_map[x] for x in from_label], dtype=np.int64)
    to_node = np.asarray([node_map[x] for x in to_label], dtype=np.int64)

    links = pd.DataFrame({
        "link_id": link_ids,
        "from_node": from_node,
        "to_node": to_node,
        "type": model_type,
        "length_miles": length_miles,
        "capacity_dry_vehph": capacity_dry,
        "capacity_flood_vehph": capacity_flood,
        "t0_dry_min": t0_dry,
        "t0_flood_min": t0_flood,
        "is_closed": is_closed.astype(np.int8),
        "station_id": station_id,
        "source_init_node_label": from_label,
        "source_term_node_label": to_label,
        "source_edge_type": source_edge_type.to_numpy(dtype=str),
        "source_flood_depth_mm": depth_mm,
        "source_flood_affected": flood_affected.astype(np.int8),
        "source_retain_in_flood_network": retained_flag.astype(np.int8),

        "source_capacity": source_capacity_raw,
        "source_capacity_raw_vehph": source_capacity_raw,
        "inferred_lanes": inferred_lanes,
        "capacity_per_lane_vehph": np.where(
            is_road, CAPACITY_PER_LANE_VEHPH, 0.0
        ),
        "modeled_capacity_dry_vehph": capacity_dry,
        "source_length_miles": length_miles,
        "source_free_flow_time_file_min": source_free_flow_time_file,

        "source_free_flow_time_min": t0_dry,
        "computed_free_flow_time_min": t0_dry,
        "source_speed_mph": source_speed_mph,
        "nominal_speed_kmh": speed_limit_kmh,
        "flood_safe_speed_kmh": safe_speed,
        "operational_speed_flood_kmh": speed_after_flood,
        "flood_capacity_retention_ratio": capacity_ratio,
        "computed_flood_free_flow_time_min": t0_flood,
    })


    road_positions = np.where(is_road)[0]
    if np.any(~np.isfinite(t0_dry[road_positions])) or np.any(
        t0_dry[road_positions] <= 0.0
    ):
        raise RuntimeError("Computed dry free-flow times are not positive for all roads.")
    if np.any(inferred_lanes[road_positions] < 1) or np.any(
        inferred_lanes[road_positions] > MAX_INFERRED_LANES
    ):
        raise RuntimeError("Inferred road lane counts are outside the 1--5 range.")
    low_capacity = is_road & (source_capacity_raw < CAPACITY_PER_LANE_VEHPH)
    high_capacity = is_road & (
        source_capacity_raw > MAX_INFERRED_LANES * CAPACITY_PER_LANE_VEHPH
    )
    if np.any(inferred_lanes[low_capacity] != 1):
        raise RuntimeError("A source capacity below 1,900 was not mapped to one lane.")
    if np.any(inferred_lanes[high_capacity] != MAX_INFERRED_LANES):
        raise RuntimeError("A source capacity above 9,500 was not capped at five lanes.")


    unaffected_retained = is_road & ~flood_affected & retained_flag
    if np.any(np.abs(capacity_flood[unaffected_retained] - capacity_dry[unaffected_retained]) > 1.0e-9):
        raise RuntimeError("Unaffected retained-road flood capacity does not equal dry capacity.")
    if np.any(np.abs(t0_flood[unaffected_retained] - t0_dry[unaffected_retained]) > 1.0e-9):
        raise RuntimeError("Unaffected retained-road flood time does not equal dry time.")
    affected_open = is_road & flood_affected & ~is_closed
    expected_capacity_flood = capacity_dry * retained_capacity_ratio(depth_mm)
    if np.any(np.abs(capacity_flood[affected_open] - expected_capacity_flood[affected_open]) > 1.0e-7):
        raise RuntimeError("Flood capacity is inconsistent with C0*kappa_C(depth).")
    expected_t0_flood = np.zeros(len(raw_net), dtype=np.float64)
    expected_t0_flood[affected_open] = (
        60.0 * length_miles[affected_open] * MILES_TO_KM
        / speed_after_flood[affected_open]
    )
    if np.any(np.abs(t0_flood[affected_open] - expected_t0_flood[affected_open]) > 1.0e-9):
        raise RuntimeError("Flood free-flow time is inconsistent with length/operational speed.")
    depth_closed = is_road & flood_affected & (depth_mm >= CLOSURE_DEPTH_MM)
    if np.any(capacity_flood[depth_closed] != 0.0) or np.any(np.isfinite(t0_flood[depth_closed])):
        raise RuntimeError("Roads at or above 300 mm were not closed consistently.")

    raw_zero_count = int(
        np.count_nonzero(is_road & (source_free_flow_time_file <= 0.0))
    )
    lane_counts = {
        lane: int(np.count_nonzero(is_road & (inferred_lanes == lane)))
        for lane in range(1, MAX_INFERRED_LANES + 1)
    }
    print(
        "Road preprocessing: dry free-flow time = "
        "60*length_mile/speed_mph; "
        f"{raw_zero_count:,} source free_flow_time values <= 0 were replaced."
    )
    print(
        "Road preprocessing: lanes = clip(ceil(source_capacity/1900),1,5); "
        + ", ".join(f"{lane}-lane={count:,}" for lane, count in lane_counts.items())
    )
    print(
        "Flood preprocessing validated: operational speed=min(nominal, safe-speed polynomial); "
        "capacity=C0*kappa_C(depth); closure depth=300 mm."
    )


    source_physical = station_source[
        station_source["node_type"].astype(str).str.strip().str.lower()
        == "charging_group"
    ].copy()
    source_physical["charging_group"] = (
        source_physical["charging_group"].astype(str).str.strip().str.upper()
    )
    source_by_station = {
        sid: group.iloc[0]
        for sid, group in source_physical.groupby("charging_group", sort=False)
    }

    topology_nodes = raw_nodes.copy()
    topology_nodes["node_type"] = (
        topology_nodes.get("node_type", "").astype(str).str.strip().str.lower()
    )
    topology_nodes["charging_group"] = (
        topology_nodes.get("charging_group", "").astype(str).str.strip().str.upper()
    )

    link_pos = {str(link_id): i for i, link_id in enumerate(link_ids)}
    station_rows: List[Dict[str, object]] = []
    for sid in STATION_IDS:
        if sid not in source_by_station:
            raise ValueError(f"Physical charging-group row for {sid} is missing.")
        source_row = source_by_station[sid]
        level2 = int(float(source_row["level2"] or 0))
        dcfc = int(float(source_row["dcfc"] or 0))
        total_ports = level2 + dcfc
        mean_service = _station_service_time_min(level2, dcfc)

        station_links = links[links["station_id"] == sid]
        index_by_type: Dict[str, int] = {}
        for typ in ("in", "out", "byp"):
            positions = station_links.index[station_links["type"] == typ].tolist()
            if len(positions) != 1:
                raise ValueError(
                    f"{sid} must have exactly one {typ!r} edge; found {len(positions)}."
                )
            index_by_type[typ] = int(positions[0])

        entry_i = index_by_type["in"]
        exit_i = index_by_type["out"]
        bypass_i = index_by_type["byp"]
        entrance_label = str(links.at[entry_i, "source_init_node_label"])
        physical_label = str(links.at[entry_i, "source_term_node_label"])
        physical_exit_label = str(links.at[exit_i, "source_init_node_label"])
        exit_label = str(links.at[exit_i, "source_term_node_label"])
        bypass_u = str(links.at[bypass_i, "source_init_node_label"])
        bypass_v = str(links.at[bypass_i, "source_term_node_label"])
        if physical_label != physical_exit_label:
            raise ValueError(f"{sid} entry and exit do not share the same physical node.")
        if (bypass_u, bypass_v) != (entrance_label, exit_label):
            raise ValueError(f"{sid} bypass does not connect entrance to exit.")

        station_rows.append({
            "station_id": sid,
            "station_node_id": node_map[physical_label],
            "k_ports": total_ports,
            "mean_service_min": mean_service,
            "cv2": 1.0,
            "level2_ports": level2,
            "dcfc_ports": dcfc,
            "entry_link_id": str(links.at[entry_i, "link_id"]),
            "exit_link_id": str(links.at[exit_i, "link_id"]),
            "bypass_link_id": str(links.at[bypass_i, "link_id"]),
            "entrance_node_id": node_map[entrance_label],
            "exit_node_id": node_map[exit_label],
            "station_node_label": physical_label,
            "entrance_node_label": entrance_label,
            "exit_node_label": exit_label,
            "service_time_formula": "(42*dcfc + 120*level2)/(dcfc+level2)",
        })

    stations = pd.DataFrame(station_rows).sort_values(
        "station_id", key=lambda s: s.str[1:].astype(int)
    ).reset_index(drop=True)
    return links, stations


def load_inputs(data_root: Path, config: ScenarioConfig) -> Tuple[NetworkData, ODData]:

    data_root = Path(data_root).expanduser()


    net_path = locate_local_csv(
        data_root, "ChicagoRegional_net_flood_kept.csv"
    )
    topology_node_path = locate_local_csv(
        data_root, "ChicagoRegional_node_flood_kept.csv"
    )


    station_source_path = topology_node_path

    non_ev_path = locate_local_csv(
        data_root, "ChicagoRegional_nonEV_flood_kept_matrix.csv"
    )
    ev_path = locate_local_csv(
        data_root, "ChicagoRegional_EV_flood_kept_matrix.csv"
    )

    print("\nReading raw local inputs directly (no prepared BIHMH CSVs):")
    print(f"  road network:      {net_path}")
    print(f"  topology nodes:    {topology_node_path}")
    print(f"  station ports:     {station_source_path}")
    print(f"  non-EV OD matrix:  {non_ev_path}")
    print(f"  EV OD matrix:      {ev_path}")

    links, stations = prepare_network_from_raw_files(
        net_path=net_path,
        topology_node_path=topology_node_path,
        station_source_path=station_source_path,
        config=config,
    )

    links = links.copy()
    links["link_id"] = links["link_id"].astype(str)
    links["type"] = links["type"].astype(str).str.strip().str.lower()
    links["station_id"] = (
        links["station_id"].fillna("").astype(str).str.strip().str.upper()
    )
    for column in [
        "from_node", "to_node", "is_closed", "length_miles",
        "capacity_dry_vehph", "capacity_flood_vehph",
        "t0_dry_min", "t0_flood_min",
    ]:
        links[column] = pd.to_numeric(links[column], errors="raise")

    node_ids = np.unique(
        np.concatenate([
            links["from_node"].astype(np.int64).to_numpy(),
            links["to_node"].astype(np.int64).to_numpy(),
        ])
    )
    node_to_index = {int(node): i for i, node in enumerate(node_ids)}
    edge_u = (
        links["from_node"].astype(np.int64).map(node_to_index).to_numpy(dtype=np.int64)
    )
    edge_v = (
        links["to_node"].astype(np.int64).map(node_to_index).to_numpy(dtype=np.int64)
    )
    edge_type = links["type"].to_numpy(dtype=str)
    edge_length_miles = links["length_miles"].to_numpy(dtype=np.float64)
    edge_t0_dry = links["t0_dry_min"].to_numpy(dtype=np.float64)
    edge_t0_flood = links["t0_flood_min"].to_numpy(dtype=np.float64)
    edge_cap_dry = links["capacity_dry_vehph"].to_numpy(dtype=np.float64)
    edge_cap_flood = links["capacity_flood_vehph"].to_numpy(dtype=np.float64)
    edge_closed = links["is_closed"].to_numpy(dtype=np.int8)
    edge_station = links["station_id"].to_numpy(dtype=str)

    station_ids = sorted(
        stations["station_id"].astype(str).str.upper().tolist(),
        key=lambda sid: int(sid[1:]),
    )
    station_row = {
        str(row.station_id).upper(): row
        for row in stations.itertuples(index=False)
    }
    operational = {sid.upper() for sid in config.operational_stations}
    unknown = sorted(operational - set(station_ids))
    if unknown:
        raise ValueError(f"Unknown operational stations: {unknown}")

    n_stations = len(station_ids)
    station_ports = np.zeros(n_stations, dtype=np.int64)
    station_service = np.zeros(n_stations, dtype=np.float64)
    station_entry_edge = np.full(n_stations, -1, dtype=np.int64)
    station_exit_edge = np.full(n_stations, -1, dtype=np.int64)
    station_bypass_edge = np.full(n_stations, -1, dtype=np.int64)
    station_entrance_node = np.full(n_stations, -1, dtype=np.int64)
    station_exit_node = np.full(n_stations, -1, dtype=np.int64)
    operational_mask = np.zeros(n_stations, dtype=np.bool_)
    link_pos = {link_id: i for i, link_id in enumerate(links["link_id"].astype(str))}

    for si, sid in enumerate(station_ids):
        row = station_row[sid]
        station_ports[si] = int(row.k_ports)
        station_service[si] = float(row.mean_service_min)
        station_entry_edge[si] = link_pos[str(row.entry_link_id)]
        station_exit_edge[si] = link_pos[str(row.exit_link_id)]
        station_bypass_edge[si] = link_pos[str(row.bypass_link_id)]
        station_entrance_node[si] = node_to_index[int(row.entrance_node_id)]
        station_exit_node[si] = node_to_index[int(row.exit_node_id)]
        operational_mask[si] = sid in operational

    is_road = edge_type == "road"
    is_byp = edge_type == "byp"
    active = np.zeros(len(links), dtype=np.bool_)
    if config.use_flood_road_state:
        retain = links["source_retain_in_flood_network"].to_numpy(dtype=np.int8) == 1
        active[is_road] = (
            retain[is_road]
            & np.isfinite(edge_t0_flood[is_road])
            & (edge_cap_flood[is_road] > 0.0)
            & (edge_closed[is_road] == 0)
        )
    else:


        active[is_road] = (
            np.isfinite(edge_t0_dry[is_road])
            & (edge_cap_dry[is_road] > 0.0)
        )
    active[is_byp] = True

    for si, sid in enumerate(station_ids):
        active[station_entry_edge[si]] = bool(operational_mask[si])
        active[station_exit_edge[si]] = bool(operational_mask[si])

        active[station_bypass_edge[si]] = True

    travel_edge = active & (is_road | is_byp)

    non_ids, non_ev = read_od_matrix(non_ev_path)
    ev_ids, ev = read_od_matrix(ev_path)
    if not np.array_equal(non_ids, ev_ids):
        raise ValueError("EV and non-EV matrix node orders differ.")
    if config.demand_scale <= 0.0:
        raise ValueError("demand_scale must be positive.")
    non_ev *= config.demand_scale
    ev *= config.demand_scale

    missing_od_nodes = [int(node) for node in non_ids if int(node) not in node_to_index]
    if missing_od_nodes:
        raise ValueError(
            f"{len(missing_od_nodes)} OD nodes are absent from the selected raw network; "
            f"first examples: {missing_od_nodes[:10]}"
        )
    od_node_index = np.asarray(
        [node_to_index[int(node)] for node in non_ids], dtype=np.int64
    )

    ev_positive = (ev.sum(axis=1) > 0.0) | (ev.sum(axis=0) > 0.0)
    ev_pos = np.where(ev_positive)[0]
    ev_node_ids = non_ids[ev_pos]
    ev_node_index = od_node_index[ev_pos]
    ev_reduced = ev[np.ix_(ev_pos, ev_pos)]

    network = NetworkData(
        links=links,
        stations=stations,
        node_ids=node_ids,
        node_to_index=node_to_index,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_type=edge_type,
        edge_length_miles=edge_length_miles,
        edge_t0_dry_min=edge_t0_dry,
        edge_t0_flood_min=edge_t0_flood,
        edge_capacity_dry=edge_cap_dry,
        edge_capacity_flood=edge_cap_flood,
        edge_is_closed=edge_closed,
        edge_station_id=edge_station,
        active_edge=active,
        travel_edge=travel_edge,
        station_ids=station_ids,
        station_ports=station_ports,
        station_service_min=station_service,
        station_entry_edge=station_entry_edge,
        station_exit_edge=station_exit_edge,
        station_bypass_edge=station_bypass_edge,
        station_entrance_node=station_entrance_node,
        station_exit_node=station_exit_node,
        operational_station_mask=operational_mask,
    )
    od = ODData(
        od_node_ids=non_ids,
        od_node_index=od_node_index,
        non_ev=non_ev,
        ev=ev_reduced,
        ev_node_ids=ev_node_ids,
        ev_node_index=ev_node_index,
        ev_od_positions=ev_pos,
        potential_non_ev=float(non_ev.sum()),
        potential_ev=float(ev_reduced.sum()),
    )

    print(
        f"Prepared in memory: {len(links):,} links, {len(node_ids):,} active-graph nodes, "
        f"{int(active.sum()):,} active edges."
    )
    print(
        "Station inputs: "
        + ", ".join(
            f"{sid}={station_ports[i]} ports/{station_service[i]:.2f} min"
            for i, sid in enumerate(station_ids)
        )
    )
    return network, od

def build_adjacency(edge_u: np.ndarray, edge_v: np.ndarray, edge_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_edges = np.where(edge_mask)[0].astype(np.int64)
    n_nodes = int(max(edge_u.max(initial=0), edge_v.max(initial=0)) + 1)

    out_order = np.argsort(edge_u[active_edges], kind="stable")
    out_edges = active_edges[out_order]
    out_counts = np.bincount(edge_u[out_edges], minlength=n_nodes)
    out_indptr = np.empty(n_nodes + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(out_counts, out=out_indptr[1:])

    in_order = np.argsort(edge_v[active_edges], kind="stable")
    in_edges = active_edges[in_order]
    in_counts = np.bincount(edge_v[in_edges], minlength=n_nodes)
    in_indptr = np.empty(n_nodes + 1, dtype=np.int64)
    in_indptr[0] = 0
    np.cumsum(in_counts, out=in_indptr[1:])
    return active_edges, out_indptr, out_edges, in_indptr, in_edges


def min_pair_sparse(
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
    edge_mask: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    best: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for e in np.where(edge_mask)[0]:
        w = float(edge_weight[e])
        if not math.isfinite(w) or w < 0.0:
            continue
        key = (int(edge_u[e]), int(edge_v[e]))
        current = best.get(key)
        if current is None or w < current[0] - 1.0e-15:
            best[key] = (w, int(e))
    if not best:
        raise ValueError("Scenario has no active travel edges.")
    pairs = sorted(best)
    rows = np.asarray([p[0] for p in pairs], dtype=np.int64)
    cols = np.asarray([p[1] for p in pairs], dtype=np.int64)
    vals = np.asarray([best[p][0] for p in pairs], dtype=np.float64)
    pair_keys = rows * np.int64(n_nodes) + cols
    pair_edges = np.asarray([best[p][1] for p in pairs], dtype=np.int64)
    matrix = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return matrix, pair_keys, pair_edges


def predecessor_nodes_to_edges(
    predecessor_node: np.ndarray,
    pair_keys: np.ndarray,
    pair_edges: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    pred = np.asarray(predecessor_node, dtype=np.int64)
    out = np.full(pred.shape, -1, dtype=np.int64)
    if pred.ndim != 2:
        raise ValueError("Expected a two-dimensional predecessor matrix.")
    node_columns = np.broadcast_to(np.arange(n_nodes, dtype=np.int64), pred.shape)
    valid = pred >= 0
    keys = pred[valid] * np.int64(n_nodes) + node_columns[valid]
    positions = np.searchsorted(pair_keys, keys)
    good = (positions < pair_keys.size) & (pair_keys[positions] == keys)
    flat = out[valid]
    flat[good] = pair_edges[positions[good]]
    out[valid] = flat
    return out


def erlang_c_wait_hours(arrivals_per_hour: float, mean_service_hours: float, servers: int) -> float:
    if servers <= 0:
        return INF
    if arrivals_per_hour <= 0.0 or mean_service_hours <= 0.0:
        return 0.0
    offered = arrivals_per_hour * mean_service_hours
    if offered >= servers:
        return INF
    rho = offered / servers
    term = 1.0
    summation = 1.0
    for n in range(1, servers):
        term *= offered / n
        summation += term
    term_k = term * offered / servers
    tail = term_k / (1.0 - rho)
    p0 = 1.0 / (summation + tail)
    lq = p0 * term_k * rho / ((1.0 - rho) ** 2)
    return lq / arrivals_per_hour


def queue_metrics(entries_period: np.ndarray, network: NetworkData, config: ScenarioConfig) -> QueueMetrics:
    n = len(network.station_ids)
    arrivals = entries_period / config.analysis_hours
    offered = arrivals * (network.station_service_min / 60.0)
    utilization = np.full(n, INF, dtype=np.float64)
    wait_h = np.full(n, INF, dtype=np.float64)
    sojourn = np.full(n, INF, dtype=np.float64)
    stable = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if not network.operational_station_mask[i]:
            continue
        k = int(network.station_ports[i])
        utilization[i] = offered[i] / k if k > 0 else INF
        if offered[i] >= k * (1.0 - 1.0e-10):
            continue
        w_mm = erlang_c_wait_hours(float(arrivals[i]), float(network.station_service_min[i] / 60.0), k)

        cv2 = 1.0
        w_mg = ((1.0 + cv2) / 2.0) * w_mm
        wait_h[i] = w_mg
        sojourn[i] = network.station_service_min[i] + 60.0 * w_mg
        stable[i] = True
    return QueueMetrics(arrivals, entries_period.copy(), offered, utilization, wait_h, sojourn, stable)


def validate_queue_choice_settings(config: ScenarioConfig) -> None:
    if config.queue_wait_disutility_multiplier < 1.0:
        raise ValueError("queue_wait_disutility_multiplier must be at least 1.0.")
    if not (0.0 <= config.queue_utilization_penalty_start < 1.0):
        raise ValueError("queue_utilization_penalty_start must be in [0, 1).")
    if config.queue_utilization_penalty_scale_min < 0.0:
        raise ValueError("queue_utilization_penalty_scale_min must be nonnegative.")
    if config.queue_utilization_penalty_power <= 0.0:
        raise ValueError("queue_utilization_penalty_power must be positive.")
    if config.queue_utilization_penalty_cap_min <= 0.0:
        raise ValueError("queue_utilization_penalty_cap_min must be positive.")
    if config.queue_unstable_loading_penalty_min <= 0.0:
        raise ValueError("queue_unstable_loading_penalty_min must be positive.")


def station_choice_costs(
    metrics: QueueMetrics,
    network: NetworkData,
    config: ScenarioConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    validate_queue_choice_settings(config)
    n = len(network.station_ids)
    choice = np.full(n, INF, dtype=np.float64)
    crowding = np.full(n, INF, dtype=np.float64)
    extra = np.full(n, INF, dtype=np.float64)
    start = float(config.queue_utilization_penalty_start)
    scale = float(config.queue_utilization_penalty_scale_min)
    power = float(config.queue_utilization_penalty_power)
    cap = float(config.queue_utilization_penalty_cap_min)
    wait_weight = float(config.queue_wait_disutility_multiplier)
    big_m = float(config.queue_unstable_loading_penalty_min)

    for i in range(n):
        if not network.operational_station_mask[i]:
            continue
        wait_h = float(metrics.average_wait_hours[i])
        util = float(metrics.utilization[i])
        if (not bool(metrics.stable[i])) or (not math.isfinite(wait_h)) or util >= 1.0:
            choice[i] = big_m
            crowding[i] = big_m
            extra[i] = big_m
            continue

        wait_min = 60.0 * max(0.0, wait_h)
        barrier = 0.0
        if util > start and scale > 0.0:

            ratio = (util - start) / max(1.0e-6, 1.0 - util)
            barrier = min(cap, scale * (ratio ** power))

        physical_sojourn = float(network.station_service_min[i]) + wait_min
        perceived = (
            float(network.station_service_min[i])
            + wait_weight * wait_min
            + barrier
        )
        choice[i] = min(big_m, perceived)
        crowding[i] = barrier
        extra[i] = max(0.0, choice[i] - physical_sojourn)

    return choice, crowding, extra


def road_times(total_flow: np.ndarray, network: NetworkData, config: ScenarioConfig) -> np.ndarray:
    n_edges = len(network.links)
    times = np.full(n_edges, INF, dtype=np.float64)
    road = network.edge_type == "road"
    byp = network.edge_type == "byp"
    times[byp & network.active_edge] = 0.0
    if config.use_flood_road_state:
        t0 = network.edge_t0_flood_min
        cap = network.edge_capacity_flood
    else:
        t0 = network.edge_t0_dry_min
        cap = network.edge_capacity_dry
    idx = np.where(road & network.active_edge)[0]
    rate = np.maximum(0.0, total_flow[idx]) / config.analysis_hours
    ratio = np.divide(rate, cap[idx], out=np.zeros_like(rate), where=cap[idx] > 0.0)
    times[idx] = t0[idx] * (1.0 + config.alpha_bpr * np.power(ratio, config.beta_bpr))
    return times


def compute_served_non_ev(non_ev: np.ndarray, finite_distances: np.ndarray, od_dest_idx: np.ndarray) -> float:
    reachable = np.isfinite(finite_distances[:, od_dest_idx])
    return float(non_ev[reachable].sum())


def build_direct_route_tree_variants(
    *,
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_type: np.ndarray,
    travel_mask: np.ndarray,
    algorithm_time: np.ndarray,
    source_nodes: np.ndarray,
    destination_nodes: np.ndarray,
    edge_length_miles: np.ndarray,
    edge_minhash: np.ndarray,
    base_time_pred_edge: np.ndarray,
    base_time_order: np.ndarray,
    distance_pred_edge: np.ndarray,
    distance_order: np.ndarray,
    candidate_count: int,
    perturbation_strength: float,
    random_seed: int,
) -> Tuple[
    Tuple[np.ndarray, ...], Tuple[np.ndarray, ...],
    np.ndarray, np.ndarray, np.ndarray
]:

    count = max(2, int(candidate_count))
    road_mask = (np.asarray(edge_type) == "road") & travel_mask
    pred_list: List[np.ndarray] = [
        np.asarray(base_time_pred_edge, dtype=np.int32).copy(),
        np.asarray(distance_pred_edge, dtype=np.int32).copy(),
    ]
    order_list: List[np.ndarray] = [
        np.asarray(base_time_order, dtype=np.int32).copy(),
        np.asarray(distance_order, dtype=np.int32).copy(),
    ]

    rng = np.random.default_rng(int(random_seed))
    for variant in range(2, count):
        weights = np.asarray(algorithm_time, dtype=np.float64).copy()


        noise = rng.random(int(np.count_nonzero(road_mask)))
        weights[road_mask] *= 1.0 + float(perturbation_strength) * noise
        sparse, pair_keys, pair_edges = min_pair_sparse(
            n_nodes, edge_u, edge_v, weights, travel_mask
        )
        dist, pred_node = dijkstra(
            sparse,
            directed=True,
            indices=source_nodes,
            return_predecessors=True,
        )
        pred_edge = predecessor_nodes_to_edges(
            pred_node, pair_keys, pair_edges, n_nodes
        ).astype(np.int32, copy=False)
        order = np.argsort(dist, axis=1).astype(np.int32)
        pred_list.append(pred_edge.copy())
        order_list.append(order.copy())
        del sparse, dist, pred_node, pred_edge, order, weights
        gc.collect()

    n_variants = len(pred_list)
    n_orig = len(source_nodes)
    n_dest = len(destination_nodes)
    n_hash = edge_minhash.shape[1]
    actual_cost = np.empty((n_variants, n_orig, n_dest), dtype=np.float64)
    miles = np.empty((n_variants, n_orig, n_dest), dtype=np.float64)
    signatures = np.empty(
        (n_variants, n_orig, n_dest, n_hash), dtype=np.uint64
    )

    for vv, (pred, order) in enumerate(zip(pred_list, order_list)):
        cost_full = cumulative_metric_on_trees(
            pred, order, source_nodes, edge_u, algorithm_time
        )
        miles_full = cumulative_metric_on_trees(
            pred, order, source_nodes, edge_u, edge_length_miles
        )
        actual_cost[vv] = cost_full[:, destination_nodes]
        miles[vv] = miles_full[:, destination_nodes]
        signatures[vv] = path_minhash_to_targets(
            pred, source_nodes, destination_nodes, edge_u, edge_minhash
        )
        del cost_full, miles_full

    return (
        tuple(pred_list), tuple(order_list), actual_cost, miles, signatures
    )


def current_direct_variant_costs(
    pool: EVPathPool,
    source_nodes: np.ndarray,
    destination_nodes: np.ndarray,
    edge_u: np.ndarray,
    algorithm_time: np.ndarray,
) -> np.ndarray:

    n_variants = len(pool.direct_variant_pred_edge)
    out = np.empty(
        (n_variants, len(source_nodes), len(destination_nodes)),
        dtype=np.float64,
    )
    for vv in range(n_variants):
        full = cumulative_metric_on_trees(
            pool.direct_variant_pred_edge[vv],
            pool.direct_variant_order[vv],
            source_nodes,
            edge_u,
            algorithm_time,
        )
        out[vv] = full[:, destination_nodes]
        del full
    return out


def perform_assignment(network: NetworkData, od: ODData, config: ScenarioConfig) -> AssignmentResult:
    if len(config.initial_soc_fractions) != len(config.ev_class_shares):
        raise ValueError("initial_soc_fractions and ev_class_shares must have equal lengths.")
    if config.ev_initial_k_paths <= 0:
        raise ValueError("ev_initial_k_paths must be positive.")
    if config.ev_path_minhash_size <= 0:
        raise ValueError("ev_path_minhash_size must be positive.")
    if config.ev_path_diversity_weight < 0.0 or config.ev_path_cost_weight < 0.0:
        raise ValueError("EV path diversity/cost weights must be nonnegative.")
    if config.ev_direct_candidate_trees < config.ev_initial_k_paths:
        raise ValueError(
            "ev_direct_candidate_trees must be at least ev_initial_k_paths "
            "so direct-feasible users can receive K non-charging candidates."
        )
    if config.ev_direct_perturbation_strength < 0.0:
        raise ValueError("ev_direct_perturbation_strength must be nonnegative.")

    shares = np.asarray(config.ev_class_shares, dtype=np.float64)
    shares /= shares.sum()
    initial_ranges = config.battery_range_miles * np.asarray(config.initial_soc_fractions, dtype=np.float64)

    n_edges = len(network.links)
    n_nodes = len(network.node_ids)
    operational_station_indices = np.where(network.operational_station_mask)[0]
    op_station_ids = [network.station_ids[i] for i in operational_station_indices]
    op_entrance_nodes = network.station_entrance_node[operational_station_indices]
    op_exit_nodes = network.station_exit_node[operational_station_indices]


    travel_mask = network.travel_edge.copy()
    _, out_indptr, out_edges, in_indptr, in_edges = build_adjacency(
        network.edge_u, network.edge_v, travel_mask
    )


    distance_weight = np.full(n_edges, INF, dtype=np.float64)
    road = network.edge_type == "road"
    byp = network.edge_type == "byp"
    distance_weight[road & travel_mask] = np.maximum(
        network.edge_length_miles[road & travel_mask], ALGORITHM_EPS_DISTANCE_MILE
    )
    distance_weight[byp & travel_mask] = ALGORITHM_EPS_DISTANCE_MILE

    ev_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)
    distance_sparse, dist_pair_keys, dist_pair_edges = min_pair_sparse(
        n_nodes, network.edge_u, network.edge_v, distance_weight, travel_mask
    )
    distance_matrix, distance_pred_node = dijkstra(
        distance_sparse,
        directed=True,
        indices=ev_sources,
        return_predecessors=True,
    )
    distance_pred_edge = predecessor_nodes_to_edges(
        distance_pred_node, dist_pair_keys, dist_pair_edges, n_nodes
    )
    n_ev_orig = len(od.ev_node_index)
    n_op_stations = len(op_station_ids)
    distance_orig = distance_matrix[:n_ev_orig]
    distance_station = distance_matrix[n_ev_orig:]
    distance_pred_orig = distance_pred_edge[:n_ev_orig]
    distance_pred_station = distance_pred_edge[n_ev_orig:]
    distance_order_orig = np.argsort(distance_orig, axis=1).astype(np.int32)
    distance_order_station = np.argsort(distance_station, axis=1).astype(np.int32)


    dest = od.ev_node_index
    edge_minhash = build_edge_minhash(
        network.edge_type,
        config.ev_path_minhash_size,
        config.ev_path_random_seed,
    )
    direct_distance_sig = path_minhash_to_targets(
        distance_pred_orig, od.ev_node_index, dest,
        network.edge_u, edge_minhash,
    )
    first_distance_sig = path_minhash_to_targets(
        distance_pred_orig, od.ev_node_index, op_entrance_nodes,
        network.edge_u, edge_minhash,
    )
    second_distance_sig = path_minhash_to_targets(
        distance_pred_station, op_exit_nodes, dest,
        network.edge_u, edge_minhash,
    )


    non_flow = np.zeros(n_edges, dtype=np.float64)
    ev_flow = np.zeros(n_edges, dtype=np.float64)
    station_entries_full = np.zeros(len(network.station_ids), dtype=np.float64)
    served_ev_by_class = np.zeros(len(shares), dtype=np.float64)
    potential_ev_by_class = od.potential_ev * shares
    served_non_ev = 0.0
    converged = False
    gap = INF
    final_link_time = np.full(n_edges, INF, dtype=np.float64)
    final_queue = queue_metrics(station_entries_full, network, config)
    ev_path_pool: Optional[EVPathPool] = None
    ev_path_pool_summary: Optional[pd.DataFrame] = None
    ev_path_pool_refreshes = 0


    non_ev_sources = od.od_node_index.astype(np.int64)
    ev_time_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)

    print(f"[{config.scenario_label}] loaded {len(network.links):,} links, {len(network.node_ids):,} nodes")
    print(f"[{config.scenario_label}] OD nodes={len(od.od_node_ids):,}; EV representative nodes={len(od.ev_node_ids):,}")
    print(f"[{config.scenario_label}] operational stations={op_station_ids}")
    print(f"[{config.scenario_label}] potential non-EV={od.potential_non_ev:.8f}; potential EV={od.potential_ev:.8f}")
    print(
        f"[{config.scenario_label}] EV route pool: K={config.ev_initial_k_paths}; "
        "direct-feasible OD--SoC classes use direct-only routes; "
        f"charging-required classes prefer distinct stations={config.ev_prefer_distinct_stations}; "
        f"refresh every {config.ev_path_refresh_every} MSA iterations"
    )
    if n_op_stations <= 1 and config.ev_prefer_distinct_stations:
        print(
            f"[{config.scenario_label}] NOTE: only {n_op_stations} charging station is operational; "
            "K routes can differ by road path, but cannot cover multiple charging stations."
        )

    stage_timing = os.environ.get("CHICAGO_STAGE_TIMING", "0") == "1"
    for iteration in range(1, config.max_msa_iterations + 1):
        _iter_clock = time.perf_counter()
        old_non = non_flow.copy()
        old_ev = ev_flow.copy()
        old_entries = station_entries_full.copy()
        total_old = old_non + old_ev
        link_time = road_times(total_old, network, config)
        qmetrics = queue_metrics(old_entries, network, config)

        algorithm_time = np.full(n_edges, INF, dtype=np.float64)
        active_travel = np.where(travel_mask)[0]
        algorithm_time[active_travel] = np.maximum(
            link_time[active_travel], ALGORITHM_EPS_TIME_MIN
        )

        time_sparse, time_pair_keys, time_pair_edges = min_pair_sparse(
            n_nodes, network.edge_u, network.edge_v, algorithm_time, travel_mask
        )
        _stage_clock = time.perf_counter()
        non_dist = dijkstra(
            time_sparse,
            directed=True,
            indices=non_ev_sources,
            return_predecessors=False,
        )

        if stage_timing:
            print(f"  iter {iteration} nonEV dijkstra: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        _stage_clock = time.perf_counter()


        non_aux, non_unassigned = dial_logit_load_all_origins(
            non_dist,
            od.non_ev,
            od.od_node_index,
            od.od_node_index,
            network.edge_u,
            network.edge_v,
            algorithm_time,
            out_indptr,
            out_edges,
            in_indptr,
            in_edges,
            config.theta_non_ev,
        )
        if stage_timing:
            print(f"  iter {iteration} nonEV Dial: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        if iteration == 1:
            served_non_ev = od.potential_non_ev - float(non_unassigned)
        del non_dist
        gc.collect()


        _stage_clock = time.perf_counter()
        ev_all_dist, ev_all_pred_node = dijkstra(
            time_sparse,
            directed=True,
            indices=ev_time_sources,
            return_predecessors=True,
        )
        if stage_timing:
            print(f"  iter {iteration} EV dijkstra: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        _stage_clock = time.perf_counter()
        ev_time_dist = ev_all_dist[:n_ev_orig]
        st_time_dist = ev_all_dist[n_ev_orig:]
        ev_pred_node = ev_all_pred_node[:n_ev_orig]
        st_pred_node = ev_all_pred_node[n_ev_orig:]
        ev_pred_edge = predecessor_nodes_to_edges(
            ev_pred_node, time_pair_keys, time_pair_edges, n_nodes
        )
        st_pred_edge = predecessor_nodes_to_edges(
            st_pred_node, time_pair_keys, time_pair_edges, n_nodes
        )
        time_order_orig = np.argsort(ev_time_dist, axis=1).astype(np.int32)
        time_order_station = np.argsort(st_time_dist, axis=1).astype(np.int32)
        time_miles_orig = cumulative_metric_on_trees(
            ev_pred_edge, time_order_orig, od.ev_node_index,
            network.edge_u, network.edge_length_miles,
        )
        time_miles_station = cumulative_metric_on_trees(
            st_pred_edge, time_order_station, op_exit_nodes,
            network.edge_u, network.edge_length_miles,
        )
        distance_tree_time_orig = cumulative_metric_on_trees(
            distance_pred_orig, distance_order_orig, od.ev_node_index,
            network.edge_u, algorithm_time,
        )
        distance_tree_time_station = cumulative_metric_on_trees(
            distance_pred_station, distance_order_station, op_exit_nodes,
            network.edge_u, algorithm_time,
        )

        direct_time_cost = ev_time_dist[:, dest]
        direct_time_miles = time_miles_orig[:, dest]
        direct_distance_cost = distance_tree_time_orig[:, dest]
        direct_shortest_miles = distance_orig[:, dest]
        first_time_cost = ev_time_dist[:, op_entrance_nodes]
        first_time_miles = time_miles_orig[:, op_entrance_nodes]
        first_distance_cost = distance_tree_time_orig[:, op_entrance_nodes]
        first_shortest_miles = distance_orig[:, op_entrance_nodes]
        second_time_cost = st_time_dist[:, dest]
        second_time_miles = time_miles_station[:, dest]
        second_distance_cost = distance_tree_time_station[:, dest]
        second_shortest_miles = distance_station[:, dest]
        station_sojourn = station_choice_costs(qmetrics, network, config)[0][operational_station_indices]

        refresh_pool = (
            ev_path_pool is None
            or (
                config.ev_path_refresh_every > 0
                and iteration % config.ev_path_refresh_every == 0
            )
        )
        if refresh_pool:
            _pool_clock = time.perf_counter()
            first_time_sig = path_minhash_to_targets(
                ev_pred_edge, od.ev_node_index, op_entrance_nodes,
                network.edge_u, edge_minhash,
            )
            second_time_sig = path_minhash_to_targets(
                st_pred_edge, op_exit_nodes, dest,
                network.edge_u, edge_minhash,
            )
            (
                direct_variant_pred_edge,
                direct_variant_order,
                direct_variant_cost,
                direct_variant_miles,
                direct_variant_sig,
            ) = build_direct_route_tree_variants(
                n_nodes=n_nodes,
                edge_u=network.edge_u,
                edge_v=network.edge_v,
                edge_type=network.edge_type,
                travel_mask=travel_mask,
                algorithm_time=algorithm_time,
                source_nodes=od.ev_node_index,
                destination_nodes=dest,
                edge_length_miles=network.edge_length_miles,
                edge_minhash=edge_minhash,
                base_time_pred_edge=ev_pred_edge,
                base_time_order=time_order_orig,
                distance_pred_edge=distance_pred_orig,
                distance_order=distance_order_orig,
                candidate_count=config.ev_direct_candidate_trees,
                perturbation_strength=config.ev_direct_perturbation_strength,
                random_seed=config.ev_path_random_seed + iteration,
            )
            (
                pool_kind, pool_station, pool_direct_tree, pool_first_tree,
                pool_second_tree, pool_count, pool_distinct_stations,
                pool_mean_jaccard_distance, pool_requires_charging,
            ) = build_diverse_ev_path_pool(
                od.ev,
                shares,
                initial_ranges,
                config.battery_range_miles,
                direct_variant_cost,
                direct_variant_miles,
                direct_shortest_miles,
                first_time_cost,
                first_time_miles,
                first_distance_cost,
                first_shortest_miles,
                second_time_cost,
                second_time_miles,
                second_distance_cost,
                second_shortest_miles,
                station_sojourn,
                direct_variant_sig,
                first_time_sig,
                first_distance_sig,
                second_time_sig,
                second_distance_sig,
                config.ev_initial_k_paths,
                config.ev_prefer_distinct_stations,
                config.ev_path_diversity_weight,
                config.ev_path_cost_weight,
                config.queue_unstable_loading_penalty_min,
            )


            direct_mask = pool_requires_charging == 0
            charging_slot = pool_kind == 1
            violation = bool(
                np.any(charging_slot & direct_mask[:, :, :, None])
            )
            if violation:
                raise RuntimeError(
                    "Direct-feasible EV path pool contains a charging route."
                )

            ev_path_pool = EVPathPool(
                kind=pool_kind,
                station=pool_station,
                direct_tree=pool_direct_tree,
                first_tree=pool_first_tree,
                second_tree=pool_second_tree,
                count=pool_count,
                distinct_station_count=pool_distinct_stations,
                mean_pairwise_jaccard_distance=pool_mean_jaccard_distance,
                requires_charging=pool_requires_charging,
                direct_variant_pred_edge=direct_variant_pred_edge,
                direct_variant_order=direct_variant_order,
                direct_variant_miles=direct_variant_miles,
                refresh_iteration=iteration,
            )
            ev_path_pool_summary = summarize_ev_path_pool(
                ev_path_pool,
                od.ev,
                shares,
                config.initial_soc_fractions,
                config.battery_range_miles,
                config.ev_initial_k_paths,
            )
            ev_path_pool_refreshes += 1
            avg_routes = float(
                np.average(
                    ev_path_pool_summary["average_number_of_routes"].fillna(0.0),
                    weights=np.maximum(
                        ev_path_pool_summary["energy_feasible_ev_demand_in_pool"],
                        1.0e-12,
                    ),
                )
            )
            avg_stations = float(
                np.average(
                    ev_path_pool_summary[
                        "average_distinct_charging_stations"
                    ].fillna(0.0),
                    weights=np.maximum(
                        ev_path_pool_summary["energy_feasible_ev_demand_in_pool"],
                        1.0e-12,
                    ),
                )
            )
            violation_demand = float(
                ev_path_pool_summary[
                    "direct_trip_pool_charging_route_violation_demand"
                ].sum()
            )
            print(
                f"[{config.scenario_label}] refreshed EV K-route pool at MSA {iteration}: "
                f"average routes={avg_routes:.3f}, "
                f"average distinct charging stations={avg_stations:.3f}, "
                f"direct-trip charging violations={violation_demand:.6f}, "
                f"build time={time.perf_counter()-_pool_clock:.2f}s"
            )
            del (
                first_time_sig, second_time_sig,
                direct_variant_cost, direct_variant_sig,
            )

        if ev_path_pool is None:
            raise RuntimeError("EV path pool was not initialized.")


        direct_variant_cost_now = current_direct_variant_costs(
            ev_path_pool,
            od.ev_node_index,
            dest,
            network.edge_u,
            algorithm_time,
        )

        if stage_timing:
            print(f"  iter {iteration} EV tree metrics/pool: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        _stage_clock = time.perf_counter()
        (
            direct_variant_q,
            first_time_q, first_dist_q,
            second_time_q, second_dist_q,
            entries_op, served_by_class,
            potential_by_class_iter, _unserved_by_class,
        ) = ev_k_path_logit_loading(
            od.ev,
            shares,
            initial_ranges,
            config.battery_range_miles,
            config.theta_ev,
            ev_path_pool.kind,
            ev_path_pool.station,
            ev_path_pool.direct_tree,
            ev_path_pool.first_tree,
            ev_path_pool.second_tree,
            ev_path_pool.count,
            direct_variant_cost_now,
            ev_path_pool.direct_variant_miles,
            first_time_cost,
            first_time_miles,
            first_distance_cost,
            first_shortest_miles,
            second_time_cost,
            second_time_miles,
            second_distance_cost,
            second_shortest_miles,
            station_sojourn,
            1.0e6,
        )

        if stage_timing:
            print(f"  iter {iteration} EV K-route choices: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        _stage_clock = time.perf_counter()
        ev_aux = np.zeros(n_edges, dtype=np.float64)
        for vv in range(direct_variant_q.shape[0]):
            ev_aux += load_direct_tree_flows(
                ev_path_pool.direct_variant_pred_edge[vv],
                ev_path_pool.direct_variant_order[vv],
                direct_variant_q[vv],
                dest,
                network.edge_u,
                n_edges,
            )

        zero_direct = np.zeros_like(od.ev)
        ev_aux += load_origin_tree_flows(
            ev_pred_edge, time_order_orig,
            zero_direct, dest,
            first_time_q, op_entrance_nodes,
            network.edge_u, n_edges,
        )
        ev_aux += load_origin_tree_flows(
            distance_pred_orig, distance_order_orig,
            zero_direct, dest,
            first_dist_q, op_entrance_nodes,
            network.edge_u, n_edges,
        )
        ev_aux += load_station_tree_flows(
            st_pred_edge, time_order_station,
            second_time_q, dest,
            network.edge_u, n_edges,
        )
        ev_aux += load_station_tree_flows(
            distance_pred_station, distance_order_station,
            second_dist_q, dest,
            network.edge_u, n_edges,
        )

        if stage_timing:
            print(f"  iter {iteration} EV tree load: {time.perf_counter()-_stage_clock:.3f}s", flush=True)
        entries_aux_full = np.zeros(len(network.station_ids), dtype=np.float64)
        entries_aux_full[operational_station_indices] = entries_op
        for local, global_si in enumerate(operational_station_indices):
            entry_e = network.station_entry_edge[global_si]
            exit_e = network.station_exit_edge[global_si]
            ev_aux[entry_e] = entries_op[local]
            ev_aux[exit_e] = entries_op[local]

        step = 1.0 / float(iteration)
        non_flow = old_non + step * (non_aux - old_non)
        ev_flow = old_ev + step * (ev_aux - old_ev)
        station_entries_full = old_entries + step * (entries_aux_full - old_entries)
        served_ev_by_class = served_by_class
        potential_ev_by_class = potential_by_class_iter

        numerator = (
            np.abs(non_flow - old_non).sum()
            + np.abs(ev_flow - old_ev).sum()
            + np.abs(station_entries_full - old_entries).sum()
        )
        denominator = max(
            1.0,
            np.abs(old_non).sum() + np.abs(old_ev).sum() + np.abs(old_entries).sum(),
        )
        gap = float(numerator / denominator)


        del (
            ev_all_dist, ev_all_pred_node, ev_time_dist, st_time_dist,
            ev_pred_node, st_pred_node, ev_pred_edge, st_pred_edge,
            time_order_orig, time_order_station, time_miles_orig, time_miles_station,
            distance_tree_time_orig, distance_tree_time_station,
            direct_time_cost, direct_time_miles, direct_distance_cost,
            direct_shortest_miles, first_time_cost, first_time_miles,
            first_distance_cost, first_shortest_miles, second_time_cost,
            second_time_miles, second_distance_cost, second_shortest_miles,
            direct_variant_q, direct_variant_cost_now, zero_direct,
            first_time_q, first_dist_q, second_time_q, second_dist_q,
            non_aux, ev_aux, time_sparse,
        )
        gc.collect()

        if stage_timing:
            print(f"  iter {iteration} total: {time.perf_counter()-_iter_clock:.3f}s", flush=True)
        if iteration == 1 or iteration % max(1, config.progress_every) == 0:
            metrics_now = queue_metrics(station_entries_full, network, config)
            compact = ", ".join(
                f"{network.station_ids[i]}:entry={station_entries_full[i]:.2f},u={metrics_now.utilization[i]:.3f}"
                for i in operational_station_indices
            )
            print(
                f"[{config.scenario_label}] MSA {iteration}: gap={gap:.6e}; "
                f"EV feasible={served_ev_by_class.sum():.6f}; {compact}"
            )

        if iteration >= config.min_msa_iterations and gap < config.msa_tolerance:
            converged = True
            break

    total_final = non_flow + ev_flow
    final_link_time = road_times(total_final, network, config)
    final_queue = queue_metrics(station_entries_full, network, config)
    return AssignmentResult(
        config=config,
        network=network,
        od=od,
        total_flow=total_final,
        ev_flow=ev_flow,
        non_ev_flow=non_flow,
        link_time_min=final_link_time,
        station_metrics=final_queue,
        served_ev_by_class=served_ev_by_class,
        potential_ev_by_class=potential_ev_by_class,
        served_non_ev=served_non_ev,
        iterations=iteration,
        converged=converged,
        relative_gap=gap,
        ev_path_pool=ev_path_pool,
        ev_path_pool_summary=ev_path_pool_summary,
        ev_path_pool_refreshes=ev_path_pool_refreshes,
    )


def finite_or_inf(value: float, digits: int = 6) -> str:
    if math.isinf(float(value)):
        return "inf"
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def build_station_table(result: AssignmentResult) -> pd.DataFrame:
    n = len(result.network.station_ids)
    rows: List[Dict[str, object]] = []
    m = result.station_metrics
    choice_cost, crowding_penalty, behavioral_extra = station_choice_costs(
        m, result.network, result.config
    )
    for i in range(n):
        entries = float(m.entries_period[i])
        wait_h = float(m.average_wait_hours[i])
        total_wait_h = 0.0 if entries <= 0.0 else (INF if math.isinf(wait_h) else entries * wait_h)
        active = bool(result.network.operational_station_mask[i])
        compact = "--"
        if active:
            compact = (
                f"{m.arrivals_per_hour[i]:.6f}|"
                f"{finite_or_inf(m.utilization[i], 6)}|"
                f"{finite_or_inf(wait_h, 6)}"
            )
        rows.append({
            "station": result.network.station_ids[i],
            "operational": active,
            "ports": int(result.network.station_ports[i]) if active else 0,
            "installed_ports": int(result.network.station_ports[i]),
            "mean_service_min": float(result.network.station_service_min[i]),
            "entries_3hr": entries,
            "arrivals_veh_per_hour": float(m.arrivals_per_hour[i]),
            "offered_load_erlangs": float(m.offered_load[i]),
            "intensity_utilization": float(m.utilization[i]),
            "queue_stable": bool(m.stable[i]),
            "average_wait_hours": wait_h,
            "average_wait_min": wait_h * 60.0 if math.isfinite(wait_h) else INF,
            "total_wait_vehicle_hours": total_wait_h,
            "total_wait_vehicle_min": total_wait_h * 60.0 if math.isfinite(total_wait_h) else INF,
            "average_service_min": float(result.network.station_service_min[i]),
            "average_sojourn_min": float(m.sojourn_min[i]),
            "physical_average_sojourn_min": float(m.sojourn_min[i]),
            "queue_wait_disutility_multiplier": float(result.config.queue_wait_disutility_multiplier),
            "utilization_crowding_penalty_min": float(crowding_penalty[i]),
            "behavioral_queue_penalty_min": float(behavioral_extra[i]),
            "perceived_station_choice_cost_min": float(choice_cost[i]),
            "compact_lambda_intensity_wait_hr": compact,
        })
    return pd.DataFrame(rows)


def build_link_table(result: AssignmentResult) -> pd.DataFrame:
    df = result.network.links.copy()
    df["active_in_scenario"] = result.network.active_edge.astype(int)
    df["total_flow_veh_per_period"] = result.total_flow
    df["ev_flow_veh_per_period"] = result.ev_flow
    df["non_ev_flow_veh_per_period"] = result.non_ev_flow
    df["travel_time_min"] = result.link_time_min
    finite_time = np.isfinite(result.link_time_min)
    df["total_vehicle_min"] = np.where(
        finite_time, result.total_flow * np.where(finite_time, result.link_time_min, 0.0),
        np.where(result.total_flow > 0.0, np.inf, 0.0),
    )
    df["ev_vehicle_min"] = np.where(
        finite_time, result.ev_flow * np.where(finite_time, result.link_time_min, 0.0),
        np.where(result.ev_flow > 0.0, np.inf, 0.0),
    )
    df["non_ev_vehicle_min"] = np.where(
        finite_time, result.non_ev_flow * np.where(finite_time, result.link_time_min, 0.0),
        np.where(result.non_ev_flow > 0.0, np.inf, 0.0),
    )
    return df


def compute_time_summary(result: AssignmentResult) -> Dict[str, float]:
    road = result.network.edge_type == "road"
    finite = np.isfinite(result.link_time_min)
    mask = road & finite
    ev_in_vehicle_min = float(np.sum(result.ev_flow[mask] * result.link_time_min[mask]))
    non_in_vehicle_min = float(np.sum(result.non_ev_flow[mask] * result.link_time_min[mask]))
    total_in_vehicle_min = ev_in_vehicle_min + non_in_vehicle_min
    served_ev = float(result.served_ev_by_class.sum())
    served_non = float(result.served_non_ev)

    station_df = build_station_table(result)
    total_wait_h_values = station_df["total_wait_vehicle_hours"].to_numpy(dtype=float)
    total_wait_h = INF if np.isinf(total_wait_h_values).any() else float(np.sum(total_wait_h_values))
    total_service_min = float(
        np.sum(
            result.station_metrics.entries_period
            * result.network.station_service_min
            * result.network.operational_station_mask.astype(float)
        )
    )
    generalized_total_min = (
        INF if math.isinf(total_wait_h)
        else total_in_vehicle_min + total_service_min + total_wait_h * 60.0
    )
    return {
        "potential_ev_travelers": float(result.od.potential_ev),
        "served_energy_feasible_ev_travelers": served_ev,
        "unserved_ev_travelers": float(result.od.potential_ev - served_ev),
        "ev_completion_rate": served_ev / result.od.potential_ev if result.od.potential_ev > 0 else float("nan"),
        "potential_non_ev_travelers": float(result.od.potential_non_ev),
        "served_non_ev_travelers": served_non,
        "unserved_non_ev_travelers": float(result.od.potential_non_ev - served_non),
        "total_served_travelers": served_ev + served_non,
        "ev_in_vehicle_time_min": ev_in_vehicle_min,
        "non_ev_in_vehicle_time_min": non_in_vehicle_min,
        "total_in_vehicle_time_min": total_in_vehicle_min,
        "ev_average_in_vehicle_time_min": ev_in_vehicle_min / served_ev if served_ev > 0 else float("nan"),
        "non_ev_average_in_vehicle_time_min": non_in_vehicle_min / served_non if served_non > 0 else float("nan"),
        "overall_average_in_vehicle_time_min": total_in_vehicle_min / (served_ev + served_non) if served_ev + served_non > 0 else float("nan"),
        "total_wait_vehicle_hours": total_wait_h,
        "total_charging_service_vehicle_min": total_service_min,
        "total_generalized_time_min": generalized_total_min,
        "msa_iterations": float(result.iterations),
        "msa_converged": float(result.converged),
        "msa_relative_gap": float(result.relative_gap),
        "ev_target_k_routes": float(result.config.ev_initial_k_paths),
        "ev_path_pool_refreshes": float(result.ev_path_pool_refreshes),
    }


def write_latex_station_table(station_df: pd.DataFrame, result: AssignmentResult, path: Path) -> None:
    label = result.config.scenario_label
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        f"Station & {label} (3-hr) \\\\",
        r"\midrule",
    ]
    for row in station_df.itertuples(index=False):
        if not row.operational:
            cell = "--"
        else:
            lam = f"{row.arrivals_veh_per_hour:.3f}"
            intensity = r"\infty" if math.isinf(row.intensity_utilization) else f"{row.intensity_utilization:.3f}"
            wait = r"\infty" if math.isinf(row.average_wait_hours) else f"{row.average_wait_hours:.3f}"
            cell = f"${lam}\\;|\\;{intensity}\\;|\\;{wait}$"
        lines.append(f"{row.station} & {cell} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Compact charging-station summary for the three-hour analysis window. "
            r"Each cell reports $\lambda\,|\,\rho\,|\,W$: arrivals (veh/h), "
            r"server utilization, and average waiting time (h).}"
        ),
        f"\\label{{tab:{result.config.scenario_id}_station_compact}}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(result: AssignmentResult, station_df: pd.DataFrame, time_summary: Mapping[str, float]) -> None:
    print("\n" + "=" * 88)
    print(f"SCENARIO: {result.config.scenario_label}")
    print("=" * 88)
    print(f"EV potential demand: {time_summary['potential_ev_travelers']:.8f}")
    print(f"EV travelers able to complete an energy-feasible trip: {time_summary['served_energy_feasible_ev_travelers']:.8f}")
    print(f"EV travelers unable to complete: {time_summary['unserved_ev_travelers']:.8f}")
    print(f"EV completion rate: {100.0 * time_summary['ev_completion_rate']:.6f}%")
    for i, frac in enumerate(result.config.initial_soc_fractions):
        print(
            f"  EV SoC {100*frac:.0f}%: potential={result.potential_ev_by_class[i]:.8f}, "
            f"completed={result.served_ev_by_class[i]:.8f}"
        )
    print(f"Non-EV potential demand: {time_summary['potential_non_ev_travelers']:.8f}")
    print(f"Non-EV completed demand: {time_summary['served_non_ev_travelers']:.8f}")
    print(f"EV in-vehicle time: {time_summary['ev_in_vehicle_time_min']:.6f} vehicle-min")
    print(f"EV average in-vehicle time: {time_summary['ev_average_in_vehicle_time_min']:.6f} min/traveler")
    print(f"Non-EV in-vehicle time: {time_summary['non_ev_in_vehicle_time_min']:.6f} vehicle-min")
    print(f"Non-EV average in-vehicle time: {time_summary['non_ev_average_in_vehicle_time_min']:.6f} min/traveler")
    print(f"Total in-vehicle time: {time_summary['total_in_vehicle_time_min']:.6f} vehicle-min")
    print(f"Overall average in-vehicle time: {time_summary['overall_average_in_vehicle_time_min']:.6f} min/traveler")
    print(f"MSA iterations: {result.iterations}; converged={result.converged}; gap={result.relative_gap:.6e}")
    print(
        f"EV K-route pool: target K={result.config.ev_initial_k_paths}; "
        f"refreshes={result.ev_path_pool_refreshes}"
    )
    if result.ev_path_pool_summary is not None:
        for row in result.ev_path_pool_summary.itertuples(index=False):
            print(
                f"  SoC {100*row.soc_fraction:.0f}%: avg routes={row.average_number_of_routes:.3f}; "
                f"direct-only avg={row.average_direct_only_routes_when_no_charge_needed:.3f}; "
                f"avg distinct stations (charging-required only)={row.average_distinct_charging_stations:.3f}; "
                f"direct-trip charging violation demand={row.direct_trip_pool_charging_route_violation_demand:.6f}"
            )
    print("\nCharging stations (entries | arrivals/h | intensity | average wait h | total wait h):")
    for row in station_df.itertuples(index=False):
        if not row.operational:
            print(f"  {row.station}: unavailable")
        else:
            print(
                f"  {row.station}: {row.entries_3hr:.6f} | {row.arrivals_veh_per_hour:.6f} | "
                f"{finite_or_inf(row.intensity_utilization, 6)} | "
                f"{finite_or_inf(row.average_wait_hours, 6)} | "
                f"{finite_or_inf(row.total_wait_vehicle_hours, 6)}"
            )


def write_outputs(result: AssignmentResult, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    station_df = build_station_table(result)
    link_df = build_link_table(result)
    time_summary = compute_time_summary(result)

    prefix = result.config.scenario_id
    link_path = output_dir / f"{prefix}_link_flows.csv"
    station_path = output_dir / f"{prefix}_station_metrics.csv"
    soc_path = output_dir / f"{prefix}_ev_completion_by_soc.csv"
    summary_csv = output_dir / f"{prefix}_summary.csv"
    summary_json = output_dir / f"{prefix}_summary.json"
    latex_path = output_dir / f"{prefix}_station_compact_table.tex"
    path_pool_summary_path = output_dir / f"{prefix}_ev_k_path_pool_summary.csv"

    link_df.to_csv(link_path, index=False)
    station_df.to_csv(station_path, index=False)
    pd.DataFrame({
        "soc_fraction": result.config.initial_soc_fractions,
        "initial_range_miles": result.config.battery_range_miles * np.asarray(result.config.initial_soc_fractions),
        "demand_share": result.config.ev_class_shares,
        "potential_ev_travelers": result.potential_ev_by_class,
        "completed_ev_travelers": result.served_ev_by_class,
        "unserved_ev_travelers": result.potential_ev_by_class - result.served_ev_by_class,
        "completion_rate": np.divide(
            result.served_ev_by_class,
            result.potential_ev_by_class,
            out=np.full_like(result.served_ev_by_class, np.nan),
            where=result.potential_ev_by_class > 0,
        ),
    }).to_csv(soc_path, index=False)
    pd.DataFrame([time_summary]).to_csv(summary_csv, index=False)
    if result.ev_path_pool_summary is not None:
        result.ev_path_pool_summary.to_csv(path_pool_summary_path, index=False)
    else:
        pd.DataFrame().to_csv(path_pool_summary_path, index=False)
    summary_payload = {
        "scenario": result.config.__dict__,
        "metrics": {k: ("inf" if math.isinf(float(v)) else v) for k, v in time_summary.items()},
        "operational_stations": list(result.config.operational_stations),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_latex_station_table(station_df, result, latex_path)

    print_summary(result, station_df, time_summary)
    if result.config.print_all_links:
        print("\nPer-link flows (all links):")
        for row in link_df.itertuples(index=False):
            print(
                f"{row.link_id}: {row.from_node}->{row.to_node}; type={row.type}; "
                f"active={row.active_in_scenario}; total={row.total_flow_veh_per_period:.8f}; "
                f"EV={row.ev_flow_veh_per_period:.8f}; nonEV={row.non_ev_flow_veh_per_period:.8f}; "
                f"time_min={finite_or_inf(row.travel_time_min, 8)}"
            )

    outputs = {
        "link_flows": link_path,
        "station_metrics": station_path,
        "ev_completion_by_soc": soc_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "latex_station_table": latex_path,
        "ev_k_path_pool_summary": path_pool_summary_path,
    }
    print("\nCreated outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


def run_scenario(config: ScenarioConfig, data_root: Path, output_dir: Path) -> AssignmentResult:
    network, od = load_inputs(data_root, config)
    result = perform_assignment(network, od, config)
    write_outputs(result, output_dir)
    return result


@dataclass
class FastTreeEVPathPool:
    kind: np.ndarray
    station: np.ndarray
    direct_tree: np.ndarray
    first_tree: np.ndarray
    second_tree: np.ndarray
    count: np.ndarray
    distinct_station_count: np.ndarray
    mean_pairwise_jaccard_distance: np.ndarray
    requires_charging: np.ndarray
    priority_station_feasible: np.ndarray
    priority_station_included: np.ndarray
    origin_variant_pred_edge: Tuple[np.ndarray, ...]
    origin_variant_order: Tuple[np.ndarray, ...]
    station_variant_pred_edge: Tuple[np.ndarray, ...]
    station_variant_order: Tuple[np.ndarray, ...]
    direct_variant_miles: np.ndarray
    first_variant_miles: np.ndarray
    second_variant_miles: np.ndarray
    refresh_iteration: int


@njit(cache=False)
def build_fast_tree_ev_path_pool(
    ev_demand: np.ndarray,
    class_shares: np.ndarray,
    initial_ranges: np.ndarray,
    full_range: float,
    direct_variant_cost: np.ndarray,
    direct_variant_miles: np.ndarray,
    direct_shortest_miles: np.ndarray,
    first_variant_cost: np.ndarray,
    first_variant_miles: np.ndarray,
    second_variant_cost: np.ndarray,
    second_variant_miles: np.ndarray,
    station_sojourn_min: np.ndarray,
    direct_variant_sig: np.ndarray,
    first_variant_sig: np.ndarray,
    second_variant_sig: np.ndarray,
    k_paths: int,
    prefer_distinct_stations: bool,
    diversity_weight: float,
    cost_weight: float,
    unstable_loading_penalty_min: float,
    priority_station_local_index: int,
    require_priority_if_feasible: bool,
    charging_pair_first: np.ndarray,
    charging_pair_second: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    n_orig, n_dest = ev_demand.shape
    n_classes = class_shares.shape[0]
    n_origin_variants = direct_variant_cost.shape[0]
    n_second_variants = second_variant_cost.shape[0]
    n_stations = station_sojourn_min.shape[0]
    n_hash = direct_variant_sig.shape[3]

    n_charging_pairs = charging_pair_first.shape[0]


    max_alt = n_origin_variants + n_stations * n_charging_pairs

    pool_kind = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_station = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int16)
    pool_direct_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_first_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_second_tree = np.full((n_classes, n_orig, n_dest, k_paths), -1, dtype=np.int8)
    pool_count = np.zeros((n_classes, n_orig, n_dest), dtype=np.int8)
    pool_distinct_stations = np.zeros((n_classes, n_orig, n_dest), dtype=np.int8)
    pool_mean_jaccard_distance = np.zeros((n_classes, n_orig, n_dest), dtype=np.float32)
    pool_requires_charging = np.zeros((n_classes, n_orig, n_dest), dtype=np.uint8)
    pool_priority_feasible = np.zeros((n_classes, n_orig, n_dest), dtype=np.uint8)
    pool_priority_included = np.zeros((n_classes, n_orig, n_dest), dtype=np.uint8)

    cand_cost = np.empty(max_alt, dtype=np.float64)
    cand_kind = np.empty(max_alt, dtype=np.int8)
    cand_station = np.empty(max_alt, dtype=np.int16)
    cand_direct_tree = np.empty(max_alt, dtype=np.int8)
    cand_first_tree = np.empty(max_alt, dtype=np.int8)
    cand_second_tree = np.empty(max_alt, dtype=np.int8)
    cand_sig = np.empty((max_alt, n_hash), dtype=np.uint64)
    selected = np.zeros(max_alt, dtype=np.uint8)
    selected_idx = np.empty(k_paths, dtype=np.int16)
    used_station = np.zeros(max(1, n_stations), dtype=np.uint8)

    for ci in range(n_classes):
        if class_shares[ci] <= 0.0:
            continue
        r0 = initial_ranges[ci]
        for oi in range(n_orig):
            for dj in range(n_dest):
                if ev_demand[oi, dj] <= 0.0:
                    continue

                direct_possible = (
                    math.isfinite(direct_shortest_miles[oi, dj])
                    and direct_shortest_miles[oi, dj] <= r0 + 1.0e-9
                )
                pool_requires_charging[ci, oi, dj] = 0 if direct_possible else 1
                n_alt = 0

                if direct_possible:

                    for vv in range(n_origin_variants):
                        route_cost = direct_variant_cost[vv, oi, dj]
                        route_miles = direct_variant_miles[vv, oi, dj]
                        if not math.isfinite(route_cost) or route_miles > r0 + 1.0e-9:
                            continue

                        duplicate = -1
                        for aa in range(n_alt):
                            same = True
                            for hh in range(n_hash):
                                if cand_sig[aa, hh] != direct_variant_sig[vv, oi, dj, hh]:
                                    same = False
                                    break
                            if same:
                                duplicate = aa
                                break
                        if duplicate >= 0:
                            if route_cost < cand_cost[duplicate]:
                                cand_cost[duplicate] = route_cost
                                cand_direct_tree[duplicate] = vv
                            continue

                        cand_cost[n_alt] = route_cost
                        cand_kind[n_alt] = 0
                        cand_station[n_alt] = -1
                        cand_direct_tree[n_alt] = vv
                        cand_first_tree[n_alt] = -1
                        cand_second_tree[n_alt] = -1
                        for hh in range(n_hash):
                            cand_sig[n_alt, hh] = direct_variant_sig[vv, oi, dj, hh]
                        n_alt += 1
                else:


                    for sj in range(n_stations):
                        for pp in range(n_charging_pairs):
                            fv = int(charging_pair_first[pp])
                            sv = int(charging_pair_second[pp])
                            if (
                                fv < 0 or fv >= n_origin_variants
                                or sv < 0 or sv >= n_second_variants
                            ):
                                continue
                            first_cost = first_variant_cost[fv, oi, sj]
                            first_miles = first_variant_miles[fv, oi, sj]
                            second_cost = second_variant_cost[sv, sj, dj]
                            second_miles = second_variant_miles[sv, sj, dj]
                            if (
                                not math.isfinite(first_cost)
                                or first_miles > r0 + 1.0e-9
                                or not math.isfinite(second_cost)
                                or second_miles > full_range + 1.0e-9
                            ):
                                continue

                            road_cost = first_cost + second_cost
                            sojourn = station_sojourn_min[sj]
                            route_cost = road_cost + (
                                sojourn if math.isfinite(sojourn)
                                else unstable_loading_penalty_min
                            )

                            duplicate = -1
                            for aa in range(n_alt):
                                if cand_station[aa] != sj:
                                    continue
                                same = True
                                for hh in range(n_hash):
                                    first_h = first_variant_sig[fv, oi, sj, hh]
                                    second_h = second_variant_sig[sv, sj, dj, hh]
                                    sig_h = first_h if first_h < second_h else second_h
                                    if cand_sig[aa, hh] != sig_h:
                                        same = False
                                        break
                                if same:
                                    duplicate = aa
                                    break
                            if duplicate >= 0:
                                if route_cost < cand_cost[duplicate]:
                                    cand_cost[duplicate] = route_cost
                                    cand_first_tree[duplicate] = fv
                                    cand_second_tree[duplicate] = sv
                                continue

                            cand_cost[n_alt] = route_cost
                            cand_kind[n_alt] = 1
                            cand_station[n_alt] = sj
                            cand_direct_tree[n_alt] = -1
                            cand_first_tree[n_alt] = fv
                            cand_second_tree[n_alt] = sv
                            for hh in range(n_hash):
                                first_h = first_variant_sig[fv, oi, sj, hh]
                                second_h = second_variant_sig[sv, sj, dj, hh]
                                cand_sig[n_alt, hh] = first_h if first_h < second_h else second_h
                            n_alt += 1

                if n_alt <= 0:
                    continue

                for aa in range(n_alt):
                    selected[aa] = 0
                for sj in range(max(1, n_stations)):
                    used_station[sj] = 0
                selected_count = 0

                cheapest_cost = np.inf
                cheapest_idx = -1
                for aa in range(n_alt):
                    if cand_cost[aa] < cheapest_cost:
                        cheapest_cost = cand_cost[aa]
                        cheapest_idx = aa


                priority_idx = -1
                priority_cost = np.inf
                if (
                    not direct_possible
                    and require_priority_if_feasible
                    and priority_station_local_index >= 0
                ):
                    for aa in range(n_alt):
                        if cand_station[aa] == priority_station_local_index:
                            pool_priority_feasible[ci, oi, dj] = 1
                            if cand_cost[aa] < priority_cost:
                                priority_cost = cand_cost[aa]
                                priority_idx = aa
                    if priority_idx >= 0 and selected_count < k_paths:
                        selected[priority_idx] = 1
                        selected_idx[selected_count] = priority_idx
                        pool_kind[ci, oi, dj, selected_count] = cand_kind[priority_idx]
                        pool_station[ci, oi, dj, selected_count] = cand_station[priority_idx]
                        pool_direct_tree[ci, oi, dj, selected_count] = cand_direct_tree[priority_idx]
                        pool_first_tree[ci, oi, dj, selected_count] = cand_first_tree[priority_idx]
                        pool_second_tree[ci, oi, dj, selected_count] = cand_second_tree[priority_idx]
                        used_station[priority_station_local_index] = 1
                        pool_priority_included[ci, oi, dj] = 1
                        selected_count += 1


                if cheapest_idx >= 0 and selected[cheapest_idx] == 0 and selected_count < k_paths:
                    selected[cheapest_idx] = 1
                    selected_idx[selected_count] = cheapest_idx
                    pool_kind[ci, oi, dj, selected_count] = cand_kind[cheapest_idx]
                    pool_station[ci, oi, dj, selected_count] = cand_station[cheapest_idx]
                    pool_direct_tree[ci, oi, dj, selected_count] = cand_direct_tree[cheapest_idx]
                    pool_first_tree[ci, oi, dj, selected_count] = cand_first_tree[cheapest_idx]
                    pool_second_tree[ci, oi, dj, selected_count] = cand_second_tree[cheapest_idx]
                    if cand_kind[cheapest_idx] == 1 and cand_station[cheapest_idx] >= 0:
                        used_station[cand_station[cheapest_idx]] = 1
                    selected_count += 1


                while selected_count < k_paths and selected_count < n_alt:
                    best_a = -1
                    best_score = -1.0e300
                    best_cost = np.inf

                    novel_station_exists = False
                    if not direct_possible and prefer_distinct_stations:
                        for aa in range(n_alt):
                            if selected[aa] == 1:
                                continue
                            sj = cand_station[aa]
                            if sj >= 0 and used_station[sj] == 0:
                                novel_station_exists = True
                                break

                    for aa in range(n_alt):
                        if selected[aa] == 1:
                            continue
                        if novel_station_exists:
                            sj = cand_station[aa]
                            if sj < 0 or used_station[sj] == 1:
                                continue

                        if selected_count == 0:
                            score = -cand_cost[aa]
                        else:
                            min_distance = 1.0
                            for bb in range(selected_count):
                                prev = selected_idx[bb]
                                matches = 0
                                for hh in range(n_hash):
                                    if cand_sig[aa, hh] == cand_sig[prev, hh]:
                                        matches += 1
                                distance = 1.0 - float(matches) / float(max(1, n_hash))
                                if distance < min_distance:
                                    min_distance = distance
                            relative_cost = (
                                (cand_cost[aa] - cheapest_cost)
                                / max(1.0, abs(cheapest_cost))
                            )
                            score = diversity_weight * min_distance - cost_weight * relative_cost

                        if score > best_score + 1.0e-12 or (
                            abs(score - best_score) <= 1.0e-12
                            and cand_cost[aa] < best_cost
                        ):
                            best_score = score
                            best_cost = cand_cost[aa]
                            best_a = aa

                    if best_a < 0:
                        break
                    selected[best_a] = 1
                    selected_idx[selected_count] = best_a
                    pool_kind[ci, oi, dj, selected_count] = cand_kind[best_a]
                    pool_station[ci, oi, dj, selected_count] = cand_station[best_a]
                    pool_direct_tree[ci, oi, dj, selected_count] = cand_direct_tree[best_a]
                    pool_first_tree[ci, oi, dj, selected_count] = cand_first_tree[best_a]
                    pool_second_tree[ci, oi, dj, selected_count] = cand_second_tree[best_a]
                    if cand_kind[best_a] == 1 and cand_station[best_a] >= 0:
                        used_station[cand_station[best_a]] = 1
                        if cand_station[best_a] == priority_station_local_index:
                            pool_priority_included[ci, oi, dj] = 1
                    selected_count += 1

                pool_count[ci, oi, dj] = selected_count
                distinct = 0
                for sj in range(n_stations):
                    if used_station[sj] == 1:
                        distinct += 1
                pool_distinct_stations[ci, oi, dj] = distinct

                if selected_count >= 2:
                    total_distance = 0.0
                    pair_count = 0
                    for aa in range(selected_count):
                        for bb in range(aa + 1, selected_count):
                            ia = selected_idx[aa]
                            ib = selected_idx[bb]
                            matches = 0
                            for hh in range(n_hash):
                                if cand_sig[ia, hh] == cand_sig[ib, hh]:
                                    matches += 1
                            total_distance += 1.0 - float(matches) / float(max(1, n_hash))
                            pair_count += 1
                    pool_mean_jaccard_distance[ci, oi, dj] = total_distance / float(pair_count)

                if (
                    pool_priority_feasible[ci, oi, dj] == 1
                    and require_priority_if_feasible
                    and pool_priority_included[ci, oi, dj] == 0
                ):


                    raise RuntimeError("Priority station feasible but not retained")

    return (
        pool_kind, pool_station, pool_direct_tree, pool_first_tree,
        pool_second_tree, pool_count, pool_distinct_stations,
        pool_mean_jaccard_distance, pool_requires_charging,
        pool_priority_feasible, pool_priority_included,
    )


@njit(cache=False)
def fast_tree_ev_logit_loading(
    ev_demand: np.ndarray,
    class_shares: np.ndarray,
    initial_ranges: np.ndarray,
    full_range: float,
    theta: float,
    pool_kind: np.ndarray,
    pool_station: np.ndarray,
    pool_direct_tree: np.ndarray,
    pool_first_tree: np.ndarray,
    pool_second_tree: np.ndarray,
    pool_count: np.ndarray,
    direct_variant_cost: np.ndarray,
    direct_variant_miles: np.ndarray,
    first_variant_cost: np.ndarray,
    first_variant_miles: np.ndarray,
    second_variant_cost: np.ndarray,
    second_variant_miles: np.ndarray,
    station_sojourn_min: np.ndarray,
    unstable_loading_penalty_min: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:

    n_orig, n_dest = ev_demand.shape
    n_classes = class_shares.shape[0]
    n_origin_variants = direct_variant_cost.shape[0]
    n_second_variants = second_variant_cost.shape[0]
    n_stations = station_sojourn_min.shape[0]
    k_paths = pool_kind.shape[3]

    direct_q = np.zeros((n_origin_variants, n_orig, n_dest), dtype=np.float64)
    first_q = np.zeros((n_origin_variants, n_orig, n_stations), dtype=np.float64)
    second_q = np.zeros((n_second_variants, n_stations, n_dest), dtype=np.float64)
    station_entries = np.zeros(n_stations, dtype=np.float64)
    served_by_class = np.zeros(n_classes, dtype=np.float64)
    potential_by_class = np.zeros(n_classes, dtype=np.float64)
    unserved_by_class = np.zeros(n_classes, dtype=np.float64)

    costs = np.empty(k_paths, dtype=np.float64)
    road_costs = np.empty(k_paths, dtype=np.float64)
    weights = np.empty(k_paths, dtype=np.float64)
    valid = np.zeros(k_paths, dtype=np.uint8)

    for ci in range(n_classes):
        share = class_shares[ci]
        r0 = initial_ranges[ci]
        for oi in range(n_orig):
            for dj in range(n_dest):
                q = ev_demand[oi, dj] * share
                if q <= 0.0:
                    continue
                potential_by_class[ci] += q
                count = int(pool_count[ci, oi, dj])
                if count <= 0:
                    unserved_by_class[ci] += q
                    continue

                n_valid = 0
                for kk in range(count):
                    valid[kk] = 0
                    kind = int(pool_kind[ci, oi, dj, kk])
                    if kind == 0:
                        vv = int(pool_direct_tree[ci, oi, dj, kk])
                        if vv < 0 or vv >= n_origin_variants:
                            continue
                        c = direct_variant_cost[vv, oi, dj]
                        miles = direct_variant_miles[vv, oi, dj]
                        if math.isfinite(c) and miles <= r0 + 1.0e-9:
                            costs[kk] = c
                            road_costs[kk] = c
                            valid[kk] = 1
                            n_valid += 1
                    elif kind == 1:
                        sj = int(pool_station[ci, oi, dj, kk])
                        fv = int(pool_first_tree[ci, oi, dj, kk])
                        sv = int(pool_second_tree[ci, oi, dj, kk])
                        if (
                            sj < 0 or sj >= n_stations
                            or fv < 0 or fv >= n_origin_variants
                            or sv < 0 or sv >= n_second_variants
                        ):
                            continue
                        c1 = first_variant_cost[fv, oi, sj]
                        m1 = first_variant_miles[fv, oi, sj]
                        c2 = second_variant_cost[sv, sj, dj]
                        m2 = second_variant_miles[sv, sj, dj]
                        if (
                            math.isfinite(c1) and math.isfinite(c2)
                            and m1 <= r0 + 1.0e-9
                            and m2 <= full_range + 1.0e-9
                        ):
                            road = c1 + c2
                            sojourn = station_sojourn_min[sj]
                            road_costs[kk] = road
                            costs[kk] = road + sojourn if math.isfinite(sojourn) else np.inf
                            valid[kk] = 1
                            n_valid += 1

                if n_valid <= 0:
                    unserved_by_class[ci] += q
                    continue
                served_by_class[ci] += q

                finite_count = 0
                cmin = np.inf
                for kk in range(count):
                    if valid[kk] == 1 and math.isfinite(costs[kk]):
                        finite_count += 1
                        if costs[kk] < cmin:
                            cmin = costs[kk]

                denom = 0.0
                if finite_count > 0:
                    for kk in range(count):
                        if valid[kk] == 1 and math.isfinite(costs[kk]):
                            exponent = -theta * (costs[kk] - cmin)
                            weights[kk] = 0.0 if exponent < -745.0 else math.exp(exponent)
                            denom += weights[kk]
                        else:
                            weights[kk] = 0.0
                else:
                    cmin = np.inf
                    for kk in range(count):
                        if valid[kk] == 1:
                            costs[kk] = road_costs[kk] + unstable_loading_penalty_min
                            if costs[kk] < cmin:
                                cmin = costs[kk]
                    for kk in range(count):
                        if valid[kk] == 1:
                            exponent = -theta * (costs[kk] - cmin)
                            weights[kk] = 0.0 if exponent < -745.0 else math.exp(exponent)
                            denom += weights[kk]
                        else:
                            weights[kk] = 0.0

                if denom <= 0.0 or not math.isfinite(denom):
                    best = -1
                    best_cost = np.inf
                    for kk in range(count):
                        if valid[kk] == 1 and costs[kk] < best_cost:
                            best_cost = costs[kk]
                            best = kk
                    if best < 0:
                        served_by_class[ci] -= q
                        unserved_by_class[ci] += q
                        continue
                    for kk in range(count):
                        weights[kk] = 1.0 if kk == best else 0.0
                    denom = 1.0

                for kk in range(count):
                    if weights[kk] <= 0.0:
                        continue
                    f = q * weights[kk] / denom
                    kind = int(pool_kind[ci, oi, dj, kk])
                    if kind == 0:
                        vv = int(pool_direct_tree[ci, oi, dj, kk])
                        direct_q[vv, oi, dj] += f
                    else:
                        sj = int(pool_station[ci, oi, dj, kk])
                        fv = int(pool_first_tree[ci, oi, dj, kk])
                        sv = int(pool_second_tree[ci, oi, dj, kk])
                        first_q[fv, oi, sj] += f
                        second_q[sv, sj, dj] += f
                        station_entries[sj] += f

    return (
        direct_q, first_q, second_q, station_entries,
        served_by_class, potential_by_class, unserved_by_class,
    )


def _current_variant_target_costs(
    pred_edges: Tuple[np.ndarray, ...],
    orders: Tuple[np.ndarray, ...],
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    edge_u: np.ndarray,
    current_edge_time: np.ndarray,
) -> np.ndarray:
    out = np.empty((len(pred_edges), len(source_nodes), len(target_nodes)), dtype=np.float64)
    for vv, (pred, order) in enumerate(zip(pred_edges, orders)):
        full = cumulative_metric_on_trees(
            pred, order, source_nodes, edge_u, current_edge_time
        )
        out[vv] = full[:, target_nodes]
        del full
    return out


def _append_priority_metrics(
    summary: pd.DataFrame,
    pool: FastTreeEVPathPool,
    ev_demand: np.ndarray,
    shares: np.ndarray,
) -> pd.DataFrame:
    result = summary.copy()
    feasible_values: List[float] = []
    included_values: List[float] = []
    rate_values: List[float] = []
    violation_values: List[float] = []
    for ci, share in enumerate(shares):
        weights = ev_demand * float(share)
        feasible = pool.priority_station_feasible[ci].astype(bool)
        included = pool.priority_station_included[ci].astype(bool)
        feasible_demand = float(weights[feasible].sum())
        included_demand = float(weights[feasible & included].sum())
        violation = float(weights[feasible & ~included].sum())
        feasible_values.append(feasible_demand)
        included_values.append(included_demand)
        rate_values.append(
            included_demand / feasible_demand if feasible_demand > 0.0 else float("nan")
        )
        violation_values.append(violation)
    result["priority_station_feasible_demand"] = feasible_values
    result["priority_station_included_demand"] = included_values
    result["priority_station_coverage_rate"] = rate_values
    result["priority_station_path_pool_violation_demand"] = violation_values
    return result


def perform_assignment(
    network: NetworkData,
    od: ODData,
    config: ScenarioConfig,
    path_cache_root: Optional[Path] = None,
) -> AssignmentResult:

    del path_cache_root
    if len(config.initial_soc_fractions) != len(config.ev_class_shares):
        raise ValueError("initial_soc_fractions and ev_class_shares must have equal lengths.")
    if config.ev_initial_k_paths <= 0:
        raise ValueError("ev_initial_k_paths must be positive.")
    if config.ev_direct_candidate_trees < config.ev_initial_k_paths:
        raise ValueError("EV_DIRECT_CANDIDATE_TREES must be at least K.")

    shares = np.asarray(config.ev_class_shares, dtype=np.float64)
    shares /= shares.sum()
    initial_ranges = config.battery_range_miles * np.asarray(
        config.initial_soc_fractions, dtype=np.float64
    )

    n_edges = len(network.links)
    n_nodes = len(network.node_ids)
    operational_station_indices = np.where(network.operational_station_mask)[0]
    op_station_ids = [network.station_ids[i] for i in operational_station_indices]
    op_entrance_nodes = network.station_entrance_node[operational_station_indices]
    op_exit_nodes = network.station_exit_node[operational_station_indices]
    n_ev_orig = len(od.ev_node_index)
    n_op_stations = len(op_station_ids)
    dest = od.ev_node_index

    priority_local = -1
    priority_id = str(config.ev_priority_station_id).strip().upper()
    for local, sid in enumerate(op_station_ids):
        if str(sid).upper() == priority_id:
            priority_local = local
            break

    travel_mask = network.travel_edge.copy()
    _, out_indptr, out_edges, in_indptr, in_edges = build_adjacency(
        network.edge_u, network.edge_v, travel_mask
    )


    distance_weight = np.full(n_edges, INF, dtype=np.float64)
    road = network.edge_type == "road"
    byp = network.edge_type == "byp"
    distance_weight[road & travel_mask] = np.maximum(
        network.edge_length_miles[road & travel_mask], ALGORITHM_EPS_DISTANCE_MILE
    )
    distance_weight[byp & travel_mask] = ALGORITHM_EPS_DISTANCE_MILE

    ev_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)
    distance_sparse, dist_pair_keys, dist_pair_edges = min_pair_sparse(
        n_nodes, network.edge_u, network.edge_v, distance_weight, travel_mask
    )
    distance_matrix, distance_pred_node = dijkstra(
        distance_sparse, directed=True, indices=ev_sources, return_predecessors=True
    )
    distance_pred_edge = predecessor_nodes_to_edges(
        distance_pred_node, dist_pair_keys, dist_pair_edges, n_nodes
    ).astype(np.int32, copy=False)
    distance_orig = distance_matrix[:n_ev_orig]
    distance_station = distance_matrix[n_ev_orig:]
    distance_pred_orig = distance_pred_edge[:n_ev_orig]
    distance_pred_station = distance_pred_edge[n_ev_orig:]
    distance_order_orig = np.argsort(distance_orig, axis=1).astype(np.int32)
    distance_order_station = np.argsort(distance_station, axis=1).astype(np.int32)
    direct_shortest_miles = distance_orig[:, dest]

    edge_minhash = build_edge_minhash(
        network.edge_type,
        config.ev_path_minhash_size,
        config.ev_path_random_seed,
    )

    non_flow = np.zeros(n_edges, dtype=np.float64)
    ev_flow = np.zeros(n_edges, dtype=np.float64)
    station_entries_full = np.zeros(len(network.station_ids), dtype=np.float64)
    served_ev_by_class = np.zeros(len(shares), dtype=np.float64)
    potential_ev_by_class = od.potential_ev * shares
    served_non_ev = 0.0
    converged = False
    gap = INF
    ev_path_pool: Optional[FastTreeEVPathPool] = None
    ev_path_pool_summary: Optional[pd.DataFrame] = None
    ev_path_pool_refreshes = 0

    non_ev_sources = od.od_node_index.astype(np.int64)
    ev_time_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)

    print(f"[{config.scenario_label}] loaded {len(network.links):,} links, {len(network.node_ids):,} nodes")
    print(f"[{config.scenario_label}] OD nodes={len(od.od_node_ids):,}; EV representative nodes={len(od.ev_node_ids):,}")
    print(f"[{config.scenario_label}] operational stations={op_station_ids}")
    print(f"[{config.scenario_label}] potential non-EV={od.potential_non_ev:.8f}; potential EV={od.potential_ev:.8f}")
    print(
        f"[{config.scenario_label}] FAST EV route pool: K={config.ev_initial_k_paths}; "
        f"tree variants={config.ev_direct_candidate_trees}; exact per-OD Yen disabled; "
        "direct-feasible classes remain direct-only."
    )
    if priority_local >= 0 and config.ev_require_priority_station_if_feasible:
        print(
            f"[{config.scenario_label}] priority-station coverage active for {priority_id}: "
            "one route is retained whenever charging is required and the station is feasible."
        )

    stage_timing = os.environ.get("CHICAGO_STAGE_TIMING", "0") == "1"
    for iteration in range(1, config.max_msa_iterations + 1):
        iter_clock = time.perf_counter()
        old_non = non_flow.copy()
        old_ev = ev_flow.copy()
        old_entries = station_entries_full.copy()
        total_old = old_non + old_ev
        link_time = road_times(total_old, network, config)
        qmetrics = queue_metrics(old_entries, network, config)

        algorithm_time = np.full(n_edges, INF, dtype=np.float64)
        active_travel = np.where(travel_mask)[0]
        algorithm_time[active_travel] = np.maximum(
            link_time[active_travel], ALGORITHM_EPS_TIME_MIN
        )
        time_sparse, time_pair_keys, time_pair_edges = min_pair_sparse(
            n_nodes, network.edge_u, network.edge_v, algorithm_time, travel_mask
        )


        non_dist = dijkstra(
            time_sparse, directed=True, indices=non_ev_sources,
            return_predecessors=False,
        )
        non_aux, non_unassigned = dial_logit_load_all_origins(
            non_dist, od.non_ev, od.od_node_index, od.od_node_index,
            network.edge_u, network.edge_v, algorithm_time,
            out_indptr, out_edges, in_indptr, in_edges, config.theta_non_ev,
        )
        if iteration == 1:
            served_non_ev = od.potential_non_ev - float(non_unassigned)
        del non_dist


        ev_all_dist, ev_all_pred_node = dijkstra(
            time_sparse, directed=True, indices=ev_time_sources,
            return_predecessors=True,
        )
        ev_pred_node = ev_all_pred_node[:n_ev_orig]
        st_pred_node = ev_all_pred_node[n_ev_orig:]
        ev_pred_edge = predecessor_nodes_to_edges(
            ev_pred_node, time_pair_keys, time_pair_edges, n_nodes
        ).astype(np.int32, copy=False)
        st_pred_edge = predecessor_nodes_to_edges(
            st_pred_node, time_pair_keys, time_pair_edges, n_nodes
        ).astype(np.int32, copy=False)
        ev_time_dist = ev_all_dist[:n_ev_orig]
        st_time_dist = ev_all_dist[n_ev_orig:]
        time_order_orig = np.argsort(ev_time_dist, axis=1).astype(np.int32)
        time_order_station = np.argsort(st_time_dist, axis=1).astype(np.int32)

        refresh_pool = (
            ev_path_pool is None
            or (
                config.ev_path_refresh_every > 0
                and iteration % config.ev_path_refresh_every == 0
            )
        )
        if refresh_pool:
            pool_clock = time.perf_counter()
            origin_targets = np.concatenate([dest, op_entrance_nodes]).astype(np.int64)
            (
                origin_preds, origin_orders, origin_cost_all,
                origin_miles_all, origin_sig_all,
            ) = build_direct_route_tree_variants(
                n_nodes=n_nodes,
                edge_u=network.edge_u,
                edge_v=network.edge_v,
                edge_type=network.edge_type,
                travel_mask=travel_mask,
                algorithm_time=algorithm_time,
                source_nodes=od.ev_node_index,
                destination_nodes=origin_targets,
                edge_length_miles=network.edge_length_miles,
                edge_minhash=edge_minhash,
                base_time_pred_edge=ev_pred_edge,
                base_time_order=time_order_orig,
                distance_pred_edge=distance_pred_orig,
                distance_order=distance_order_orig,
                candidate_count=config.ev_direct_candidate_trees,
                perturbation_strength=config.ev_direct_perturbation_strength,
                random_seed=config.ev_path_random_seed + iteration,
            )
            (
                station_preds, station_orders, second_cost,
                second_miles, second_sig,
            ) = build_direct_route_tree_variants(
                n_nodes=n_nodes,
                edge_u=network.edge_u,
                edge_v=network.edge_v,
                edge_type=network.edge_type,
                travel_mask=travel_mask,
                algorithm_time=algorithm_time,
                source_nodes=op_exit_nodes,
                destination_nodes=dest,
                edge_length_miles=network.edge_length_miles,
                edge_minhash=edge_minhash,
                base_time_pred_edge=st_pred_edge,
                base_time_order=time_order_station,
                distance_pred_edge=distance_pred_station,
                distance_order=distance_order_station,
                candidate_count=config.ev_direct_candidate_trees,
                perturbation_strength=config.ev_direct_perturbation_strength,
                random_seed=config.ev_path_random_seed + 100000 + iteration,
            )

            n_dest = len(dest)
            direct_cost = origin_cost_all[:, :, :n_dest]
            direct_miles = origin_miles_all[:, :, :n_dest]
            direct_sig = origin_sig_all[:, :, :n_dest, :]
            first_cost = origin_cost_all[:, :, n_dest:]
            first_miles = origin_miles_all[:, :, n_dest:]
            first_sig = origin_sig_all[:, :, n_dest:, :]
            station_sojourn = station_choice_costs(qmetrics, network, config)[0][operational_station_indices]


            n_origin_variants = direct_cost.shape[0]
            n_station_variants = second_cost.shape[0]
            pair_first: List[int] = []
            pair_second: List[int] = []
            for vv in range(min(n_origin_variants, n_station_variants)):
                pair_first.append(vv); pair_second.append(vv)
            for vv in range(1, n_origin_variants):
                pair_first.append(vv); pair_second.append(0)
            for vv in range(1, n_station_variants):
                pair_first.append(0); pair_second.append(vv)
            pair_first_arr = np.asarray(pair_first, dtype=np.int8)
            pair_second_arr = np.asarray(pair_second, dtype=np.int8)

            (
                pool_kind, pool_station, pool_direct_tree, pool_first_tree,
                pool_second_tree, pool_count, pool_distinct_stations,
                pool_mean_jaccard, pool_requires_charging,
                pool_priority_feasible, pool_priority_included,
            ) = build_fast_tree_ev_path_pool(
                od.ev, shares, initial_ranges, config.battery_range_miles,
                direct_cost, direct_miles, direct_shortest_miles,
                first_cost, first_miles, second_cost, second_miles,
                station_sojourn, direct_sig, first_sig, second_sig,
                config.ev_initial_k_paths,
                config.ev_prefer_distinct_stations,
                config.ev_path_diversity_weight,
                config.ev_path_cost_weight,
                config.queue_unstable_loading_penalty_min,
                int(priority_local),
                bool(config.ev_require_priority_station_if_feasible),
                pair_first_arr, pair_second_arr,
            )

            direct_mask = pool_requires_charging == 0
            if bool(np.any((pool_kind == 1) & direct_mask[:, :, :, None])):
                raise RuntimeError("Direct-feasible EV path pool contains a charging route.")
            if bool(np.any((pool_priority_feasible == 1) & (pool_priority_included == 0))):
                raise RuntimeError("Priority station was feasible but absent from the route pool.")

            ev_path_pool = FastTreeEVPathPool(
                kind=pool_kind,
                station=pool_station,
                direct_tree=pool_direct_tree,
                first_tree=pool_first_tree,
                second_tree=pool_second_tree,
                count=pool_count,
                distinct_station_count=pool_distinct_stations,
                mean_pairwise_jaccard_distance=pool_mean_jaccard,
                requires_charging=pool_requires_charging,
                priority_station_feasible=pool_priority_feasible,
                priority_station_included=pool_priority_included,
                origin_variant_pred_edge=origin_preds,
                origin_variant_order=origin_orders,
                station_variant_pred_edge=station_preds,
                station_variant_order=station_orders,
                direct_variant_miles=direct_miles,
                first_variant_miles=first_miles,
                second_variant_miles=second_miles,
                refresh_iteration=iteration,
            )
            ev_path_pool_summary = summarize_ev_path_pool(
                ev_path_pool, od.ev, shares,
                config.initial_soc_fractions,
                config.battery_range_miles,
                config.ev_initial_k_paths,
            )
            ev_path_pool_summary = _append_priority_metrics(
                ev_path_pool_summary, ev_path_pool, od.ev, shares
            )
            ev_path_pool_refreshes += 1
            print(
                f"[{config.scenario_label}] built FAST K-route pool at MSA {iteration}: "
                f"variants={len(origin_preds)}, build time={time.perf_counter()-pool_clock:.2f}s"
            )
            del origin_cost_all, origin_miles_all, origin_sig_all
            del direct_cost, first_cost, second_cost, direct_sig, first_sig, second_sig
            del pair_first_arr, pair_second_arr

        if ev_path_pool is None:
            raise RuntimeError("Fast EV route pool was not initialized.")

        origin_targets = np.concatenate([dest, op_entrance_nodes]).astype(np.int64)
        origin_current = _current_variant_target_costs(
            ev_path_pool.origin_variant_pred_edge,
            ev_path_pool.origin_variant_order,
            od.ev_node_index,
            origin_targets,
            network.edge_u,
            algorithm_time,
        )
        station_current = _current_variant_target_costs(
            ev_path_pool.station_variant_pred_edge,
            ev_path_pool.station_variant_order,
            op_exit_nodes,
            dest,
            network.edge_u,
            algorithm_time,
        )
        n_dest = len(dest)
        direct_current = origin_current[:, :, :n_dest]
        first_current = origin_current[:, :, n_dest:]
        station_sojourn = station_choice_costs(qmetrics, network, config)[0][operational_station_indices]

        (
            direct_q, first_q, second_q, entries_op,
            served_by_class, potential_by_class_iter, _unserved_by_class,
        ) = fast_tree_ev_logit_loading(
            od.ev, shares, initial_ranges, config.battery_range_miles,
            config.theta_ev,
            ev_path_pool.kind, ev_path_pool.station,
            ev_path_pool.direct_tree, ev_path_pool.first_tree,
            ev_path_pool.second_tree, ev_path_pool.count,
            direct_current, ev_path_pool.direct_variant_miles,
            first_current, ev_path_pool.first_variant_miles,
            station_current, ev_path_pool.second_variant_miles,
            station_sojourn, config.queue_unstable_loading_penalty_min,
        )

        ev_aux = np.zeros(n_edges, dtype=np.float64)
        for vv in range(direct_q.shape[0]):
            ev_aux += load_origin_tree_flows(
                ev_path_pool.origin_variant_pred_edge[vv],
                ev_path_pool.origin_variant_order[vv],
                direct_q[vv], dest,
                first_q[vv], op_entrance_nodes,
                network.edge_u, n_edges,
            )
        for vv in range(second_q.shape[0]):
            ev_aux += load_station_tree_flows(
                ev_path_pool.station_variant_pred_edge[vv],
                ev_path_pool.station_variant_order[vv],
                second_q[vv], dest,
                network.edge_u, n_edges,
            )

        entries_aux_full = np.zeros(len(network.station_ids), dtype=np.float64)
        entries_aux_full[operational_station_indices] = entries_op
        for local, global_si in enumerate(operational_station_indices):
            ev_aux[network.station_entry_edge[global_si]] = entries_op[local]
            ev_aux[network.station_exit_edge[global_si]] = entries_op[local]

        step = 1.0 / float(iteration)
        non_flow = old_non + step * (non_aux - old_non)
        ev_flow = old_ev + step * (ev_aux - old_ev)
        station_entries_full = old_entries + step * (entries_aux_full - old_entries)
        served_ev_by_class = served_by_class
        potential_ev_by_class = potential_by_class_iter

        numerator = (
            np.abs(non_flow - old_non).sum()
            + np.abs(ev_flow - old_ev).sum()
            + np.abs(station_entries_full - old_entries).sum()
        )
        denominator = max(
            1.0,
            np.abs(old_non).sum() + np.abs(old_ev).sum() + np.abs(old_entries).sum(),
        )
        gap = float(numerator / denominator)

        del non_aux, ev_aux, time_sparse, ev_all_dist, ev_all_pred_node
        del ev_pred_node, st_pred_node, ev_pred_edge, st_pred_edge
        del ev_time_dist, st_time_dist, time_order_orig, time_order_station
        del origin_current, station_current, direct_current, first_current
        del direct_q, first_q, second_q
        gc.collect()

        if stage_timing:
            print(f"  iter {iteration} total: {time.perf_counter()-iter_clock:.3f}s", flush=True)
        if iteration == 1 or iteration % max(1, config.progress_every) == 0:
            metrics_now = queue_metrics(station_entries_full, network, config)
            compact = ", ".join(
                f"{network.station_ids[i]}:entry={station_entries_full[i]:.2f},u={metrics_now.utilization[i]:.3f}"
                for i in operational_station_indices
            )
            print(
                f"[{config.scenario_label}] MSA {iteration}: gap={gap:.6e}; "
                f"EV feasible={served_ev_by_class.sum():.6f}; {compact}"
            )
        if iteration >= config.min_msa_iterations and gap < config.msa_tolerance:
            converged = True
            break

    total_final = non_flow + ev_flow
    final_link_time = road_times(total_final, network, config)
    final_queue = queue_metrics(station_entries_full, network, config)
    return AssignmentResult(
        config=config,
        network=network,
        od=od,
        total_flow=total_final,
        ev_flow=ev_flow,
        non_ev_flow=non_flow,
        link_time_min=final_link_time,
        station_metrics=final_queue,
        served_ev_by_class=served_ev_by_class,
        potential_ev_by_class=potential_ev_by_class,
        served_non_ev=served_non_ev,
        iterations=iteration,
        converged=converged,
        relative_gap=gap,
        ev_path_pool=ev_path_pool,
        ev_path_pool_summary=ev_path_pool_summary,
        ev_path_pool_refreshes=ev_path_pool_refreshes,
    )


def run_scenario(config: ScenarioConfig, data_root: Path, output_dir: Path) -> AssignmentResult:
    network, od = load_inputs(data_root, config)
    result = perform_assignment(network, od, config)
    write_outputs(result, output_dir)
    return result


from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from scipy.sparse.csgraph import yen as scipy_yen


@dataclass(frozen=True)
class HybridYenSettings:
    enabled: bool = True
    initial_max_records: int = 1000
    cumulative_demand_coverage: float = 0.85
    search_k: int = 8
    overlap_trigger: float = 0.80
    minimum_station_choices: int = 2
    corridor_expansions: Tuple[int, ...] = (0, 1, 2)
    allow_full_graph_fallback: bool = False
    max_stations_per_record: int = 3
    n_jobs: int = 8
    progress_every: int = 100
    enrich_every: int = 20
    enrich_top_m: int = 50
    cache_initial_pool: bool = True
    rebuild_cache: bool = False


HYBRID_YEN_SETTINGS = HybridYenSettings()


@dataclass(frozen=True)
class HybridCandidatePath:
    nodes: np.ndarray
    edges: np.ndarray
    base_cost_min: float
    distance_miles: float
    station_local_index: int = -1


@dataclass
class HybridExplicitPathPool:
    record_class: np.ndarray
    record_origin: np.ndarray
    record_dest: np.ndarray
    record_demand: np.ndarray
    record_path_offsets: np.ndarray
    path_station: np.ndarray
    path_edge_offsets: np.ndarray
    path_edges: np.ndarray
    record_round: np.ndarray
    record_reason: np.ndarray

    @property
    def n_records(self) -> int:
        return int(self.record_demand.size)

    @property
    def n_paths(self) -> int:
        return int(self.path_station.size)

    @property
    def n_path_edges(self) -> int:
        return int(self.path_edges.size)


@dataclass
class HybridRefinementState:
    records: Dict[Tuple[int, int, int], List[HybridCandidatePath]]
    reasons: Dict[Tuple[int, int, int], str]
    rounds: Dict[Tuple[int, int, int], int]
    compact: HybridExplicitPathPool
    mask: np.ndarray
    initial_build_seconds: float = 0.0
    dynamic_build_seconds: float = 0.0
    enrichment_rounds: int = 0


@dataclass
class _HybridSparseGraph:
    matrix: csr_matrix
    pair_keys: np.ndarray
    pair_edges: np.ndarray
    edge_time_min: np.ndarray
    edge_length_miles: np.ndarray
    n_nodes: int


def _hybrid_build_sparse_graph(
    network: NetworkData,
    rank_weight: np.ndarray,
    edge_time_min: np.ndarray,
    edge_mask: np.ndarray,
) -> _HybridSparseGraph:
    matrix, pair_keys, pair_edges = min_pair_sparse(
        len(network.node_ids),
        network.edge_u,
        network.edge_v,
        rank_weight,
        edge_mask,
    )
    return _HybridSparseGraph(
        matrix=matrix,
        pair_keys=pair_keys,
        pair_edges=pair_edges,
        edge_time_min=np.asarray(edge_time_min, dtype=np.float64),
        edge_length_miles=np.asarray(network.edge_length_miles, dtype=np.float64),
        n_nodes=len(network.node_ids),
    )


def _hybrid_edge_for_pair(graph: _HybridSparseGraph, u: int, v: int) -> int:
    key = np.int64(u) * np.int64(graph.n_nodes) + np.int64(v)
    position = int(np.searchsorted(graph.pair_keys, key))
    if position >= graph.pair_keys.size or graph.pair_keys[position] != key:
        return -1
    return int(graph.pair_edges[position])


def _hybrid_reconstruct_yen_path(
    graph: _HybridSparseGraph,
    predecessor_row: np.ndarray,
    source: int,
    sink: int,
    station_local_index: int = -1,
) -> Optional[HybridCandidatePath]:
    current = int(sink)
    nodes: List[int] = [current]
    edges: List[int] = []
    seen = {current}
    while current != int(source) and len(nodes) <= graph.n_nodes:
        previous = int(predecessor_row[current])
        if previous < 0 or previous in seen:
            return None
        edge = _hybrid_edge_for_pair(graph, previous, current)
        if edge < 0:
            return None
        edges.append(edge)
        nodes.append(previous)
        seen.add(previous)
        current = previous
    if current != int(source):
        return None
    nodes.reverse()
    edges.reverse()
    edge_array = np.asarray(edges, dtype=np.int32)
    return HybridCandidatePath(
        nodes=np.asarray(nodes, dtype=np.int32),
        edges=edge_array,
        base_cost_min=float(graph.edge_time_min[edge_array].sum()) if edge_array.size else 0.0,
        distance_miles=float(graph.edge_length_miles[edge_array].sum()) if edge_array.size else 0.0,
        station_local_index=int(station_local_index),
    )


def _hybrid_scipy_yen_paths(
    graph: _HybridSparseGraph,
    source: int,
    sink: int,
    k_search: int,
) -> List[HybridCandidatePath]:
    if int(source) == int(sink) or int(k_search) <= 0:
        return []
    try:
        _, predecessors = scipy_yen(
            graph.matrix,
            int(source),
            int(sink),
            int(k_search),
            directed=True,
            return_predecessors=True,
        )
    except Exception:
        return []
    result: List[HybridCandidatePath] = []
    seen: set[Tuple[int, ...]] = set()
    for row in np.atleast_2d(predecessors):
        path = _hybrid_reconstruct_yen_path(graph, row, source, sink)
        if path is None or path.edges.size == 0:
            continue
        key = tuple(int(edge) for edge in path.edges)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _hybrid_reconstruct_tree_path(
    predecessor_edges: np.ndarray,
    source: int,
    target: int,
    network: NetworkData,
    edge_time: np.ndarray,
    station_local_index: int = -1,
) -> Optional[HybridCandidatePath]:
    current = int(target)
    source = int(source)
    edges: List[int] = []
    nodes: List[int] = [current]
    seen = {current}
    while current != source and len(nodes) <= len(network.node_ids):
        edge = int(predecessor_edges[current])
        if edge < 0:
            return None
        previous = int(network.edge_u[edge])
        if previous in seen:
            return None
        edges.append(edge)
        nodes.append(previous)
        seen.add(previous)
        current = previous
    if current != source:
        return None
    edges.reverse()
    nodes.reverse()
    edge_array = np.asarray(edges, dtype=np.int32)
    return HybridCandidatePath(
        nodes=np.asarray(nodes, dtype=np.int32),
        edges=edge_array,
        base_cost_min=float(edge_time[edge_array].sum()) if edge_array.size else 0.0,
        distance_miles=float(network.edge_length_miles[edge_array].sum()) if edge_array.size else 0.0,
        station_local_index=int(station_local_index),
    )


def _hybrid_merge_unique_paths(
    *groups: Sequence[HybridCandidatePath],
) -> List[HybridCandidatePath]:
    unique: Dict[Tuple[int, ...], HybridCandidatePath] = {}
    for group in groups:
        for path in group:
            key = tuple(int(edge) for edge in path.edges)
            previous = unique.get(key)
            if previous is None or path.base_cost_min < previous.base_cost_min:
                unique[key] = path
    return sorted(
        unique.values(),
        key=lambda path: (path.base_cost_min, path.distance_miles),
    )


def _hybrid_road_edge_set(
    path: HybridCandidatePath,
    road_edge_mask: np.ndarray,
) -> set[int]:
    return {int(edge) for edge in path.edges if bool(road_edge_mask[int(edge)])}


def _hybrid_jaccard_distance(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def _hybrid_select_low_overlap(
    candidates: Sequence[HybridCandidatePath],
    k_target: int,
    road_edge_mask: np.ndarray,
    diversity_weight: float,
    cost_weight: float,
) -> List[HybridCandidatePath]:
    if not candidates or k_target <= 0:
        return []
    remaining = sorted(
        _hybrid_merge_unique_paths(candidates),
        key=lambda path: (path.base_cost_min, path.distance_miles),
    )
    selected = [remaining.pop(0)]
    selected_sets = [_hybrid_road_edge_set(selected[0], road_edge_mask)]
    reference = max(1.0, abs(selected[0].base_cost_min))
    while remaining and len(selected) < k_target:
        best_index = -1
        best_score = -1.0e300
        best_cost = np.inf
        for index, path in enumerate(remaining):
            path_set = _hybrid_road_edge_set(path, road_edge_mask)
            minimum_distance = min(
                _hybrid_jaccard_distance(path_set, existing)
                for existing in selected_sets
            )
            relative_cost = (path.base_cost_min - selected[0].base_cost_min) / reference
            score = diversity_weight * minimum_distance - cost_weight * relative_cost
            if score > best_score + 1.0e-12 or (
                abs(score - best_score) <= 1.0e-12
                and path.base_cost_min < best_cost
            ):
                best_index = index
                best_score = score
                best_cost = path.base_cost_min
        if best_index < 0:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        selected_sets.append(_hybrid_road_edge_set(chosen, road_edge_mask))
    return selected


def _hybrid_select_station_diverse(
    candidates_by_station: Mapping[int, Sequence[HybridCandidatePath]],
    k_target: int,
    road_edge_mask: np.ndarray,
    diversity_weight: float,
    cost_weight: float,
    priority_station_local: int,
    require_priority_if_feasible: bool,
) -> List[HybridCandidatePath]:
    nonempty = {
        int(station): sorted(
            _hybrid_merge_unique_paths(paths),
            key=lambda path: (path.base_cost_min, path.distance_miles),
        )
        for station, paths in candidates_by_station.items()
        if paths
    }
    if not nonempty or k_target <= 0:
        return []
    selected: List[HybridCandidatePath] = []
    selected_keys: set[Tuple[int, ...]] = set()

    def add(path: HybridCandidatePath) -> None:
        if len(selected) >= k_target:
            return
        key = tuple(int(edge) for edge in path.edges)
        if key not in selected_keys:
            selected.append(path)
            selected_keys.add(key)

    priority_feasible = (
        bool(require_priority_if_feasible)
        and int(priority_station_local) in nonempty
    )
    if priority_feasible:
        add(nonempty[int(priority_station_local)][0])

    representatives = [paths[0] for paths in nonempty.values()]
    add(min(representatives, key=lambda path: (path.base_cost_min, path.distance_miles)))


    covered = {int(path.station_local_index) for path in selected}
    uncovered = [
        paths[0]
        for station, paths in nonempty.items()
        if station not in covered
    ]
    while uncovered and len(selected) < k_target:
        if not selected:
            add(uncovered.pop(0))
            continue
        selected_sets = [_hybrid_road_edge_set(path, road_edge_mask) for path in selected]
        best_index = max(
            range(len(uncovered)),
            key=lambda index: min(
                _hybrid_jaccard_distance(
                    _hybrid_road_edge_set(uncovered[index], road_edge_mask),
                    existing,
                )
                for existing in selected_sets
            ),
        )
        add(uncovered.pop(best_index))

    remaining = [
        path
        for paths in nonempty.values()
        for path in paths
        if tuple(int(edge) for edge in path.edges) not in selected_keys
    ]
    if remaining and len(selected) < k_target:
        merged = _hybrid_select_low_overlap(
            list(selected) + remaining,
            k_target,
            road_edge_mask,
            diversity_weight,
            cost_weight,
        )
        selected = merged[:k_target]

    if priority_feasible and not any(
        int(path.station_local_index) == int(priority_station_local)
        for path in selected
    ):
        raise RuntimeError(
            "Priority station was feasible but absent after hybrid Yen refinement."
        )
    return selected[:k_target]


def _hybrid_combined_path_is_loopless(
    first: HybridCandidatePath,
    second: HybridCandidatePath,
    station_physical_node: int,
) -> bool:
    first_nodes = {int(node) for node in first.nodes}
    second_nodes = {int(node) for node in second.nodes}
    return (
        not first_nodes.intersection(second_nodes)
        and int(station_physical_node) not in first_nodes
        and int(station_physical_node) not in second_nodes
    )


def _hybrid_combine_station_paths(
    first_paths: Sequence[HybridCandidatePath],
    second_paths: Sequence[HybridCandidatePath],
    station_local: int,
    station_global: int,
    network: NetworkData,
    station_sojourn_min: float,
    k_target: int,
    road_edge_mask: np.ndarray,
    diversity_weight: float,
    cost_weight: float,
) -> List[HybridCandidatePath]:
    entry_edge = int(network.station_entry_edge[station_global])
    exit_edge = int(network.station_exit_edge[station_global])
    station_node = int(network.edge_v[entry_edge])
    candidates: List[HybridCandidatePath] = []
    seen: set[Tuple[int, ...]] = set()
    for first in first_paths:
        for second in second_paths:
            if not _hybrid_combined_path_is_loopless(first, second, station_node):
                continue
            edge_tuple = (
                tuple(int(edge) for edge in first.edges)
                + (entry_edge, exit_edge)
                + tuple(int(edge) for edge in second.edges)
            )
            if edge_tuple in seen:
                continue
            seen.add(edge_tuple)
            nodes = np.concatenate([
                first.nodes,
                np.asarray([station_node], dtype=np.int32),
                second.nodes,
            ])
            candidates.append(HybridCandidatePath(
                nodes=nodes,
                edges=np.asarray(edge_tuple, dtype=np.int32),
                base_cost_min=(
                    first.base_cost_min
                    + float(station_sojourn_min)
                    + second.base_cost_min
                ),
                distance_miles=first.distance_miles + second.distance_miles,
                station_local_index=int(station_local),
            ))
    return _hybrid_select_low_overlap(
        candidates,
        k_target,
        road_edge_mask,
        diversity_weight,
        cost_weight,
    )


def _hybrid_corridor_node_mask(
    seed_paths: Sequence[HybridCandidatePath],
    source: int,
    sink: int,
    network: NetworkData,
    travel_mask: np.ndarray,
    expansion_steps: int,
) -> np.ndarray:
    node_mask = np.zeros(len(network.node_ids), dtype=bool)
    node_mask[int(source)] = True
    node_mask[int(sink)] = True
    for path in seed_paths:
        node_mask[path.nodes.astype(np.int64)] = True
    for _ in range(max(0, int(expansion_steps))):
        touching = travel_mask & (
            node_mask[network.edge_u] | node_mask[network.edge_v]
        )
        if not bool(np.any(touching)):
            break
        node_mask[network.edge_u[touching]] = True
        node_mask[network.edge_v[touching]] = True
    return node_mask


def _hybrid_adaptive_corridor_yen(
    *,
    seed_paths: Sequence[HybridCandidatePath],
    source: int,
    sink: int,
    max_distance_miles: float,
    network: NetworkData,
    travel_mask: np.ndarray,
    current_edge_time: np.ndarray,
    distance_rank_weight: np.ndarray,
    settings: HybridYenSettings,
    k_target: int,
    road_edge_mask: np.ndarray,
    diversity_weight: float,
    cost_weight: float,
) -> List[HybridCandidatePath]:
    accumulated = list(seed_paths)
    expansions = tuple(int(value) for value in settings.corridor_expansions)
    for expansion in expansions:
        if not seed_paths and not settings.allow_full_graph_fallback:
            break
        if seed_paths:
            node_mask = _hybrid_corridor_node_mask(
                seed_paths, source, sink, network, travel_mask, expansion
            )
            mask = (
                travel_mask
                & node_mask[network.edge_u]
                & node_mask[network.edge_v]
            )
        else:
            mask = travel_mask.copy()
        if not bool(np.any(mask)):
            continue
        time_rank = np.full(len(network.links), INF, dtype=np.float64)
        time_rank[mask] = np.maximum(
            current_edge_time[mask], ALGORITHM_EPS_TIME_MIN
        )
        distance_rank = np.full(len(network.links), INF, dtype=np.float64)
        distance_rank[mask] = distance_rank_weight[mask]
        try:
            time_graph = _hybrid_build_sparse_graph(
                network, time_rank, current_edge_time, mask
            )
            distance_graph = _hybrid_build_sparse_graph(
                network, distance_rank, current_edge_time, mask
            )
        except Exception:
            continue
        time_paths = [
            path for path in _hybrid_scipy_yen_paths(
                time_graph, source, sink, settings.search_k
            )
            if path.distance_miles <= float(max_distance_miles) + 1.0e-9
        ]
        distance_paths: List[HybridCandidatePath] = []
        if len(time_paths) < k_target:
            distance_paths = [
                path for path in _hybrid_scipy_yen_paths(
                    distance_graph, source, sink, settings.search_k
                )
                if path.distance_miles <= float(max_distance_miles) + 1.0e-9
            ]
        accumulated = _hybrid_merge_unique_paths(
            accumulated, time_paths, distance_paths
        )
        selected = _hybrid_select_low_overlap(
            accumulated,
            k_target,
            road_edge_mask,
            diversity_weight,
            cost_weight,
        )
        if len(selected) >= k_target:
            return selected

    if settings.allow_full_graph_fallback and len(accumulated) < k_target:
        mask = travel_mask.copy()
        time_rank = np.full(len(network.links), INF, dtype=np.float64)
        time_rank[mask] = np.maximum(current_edge_time[mask], ALGORITHM_EPS_TIME_MIN)
        distance_rank = np.full(len(network.links), INF, dtype=np.float64)
        distance_rank[mask] = distance_rank_weight[mask]
        try:
            time_graph = _hybrid_build_sparse_graph(network, time_rank, current_edge_time, mask)
            distance_graph = _hybrid_build_sparse_graph(network, distance_rank, current_edge_time, mask)
            accumulated = _hybrid_merge_unique_paths(
                accumulated,
                [
                    path for path in _hybrid_scipy_yen_paths(
                        time_graph, source, sink, settings.search_k
                    )
                    if path.distance_miles <= max_distance_miles + 1.0e-9
                ],
                [
                    path for path in _hybrid_scipy_yen_paths(
                        distance_graph, source, sink, settings.search_k
                    )
                    if path.distance_miles <= max_distance_miles + 1.0e-9
                ],
            )
        except Exception:
            pass
    return _hybrid_select_low_overlap(
        accumulated,
        k_target,
        road_edge_mask,
        diversity_weight,
        cost_weight,
    )


def _hybrid_fast_seed_paths_for_record(
    key: Tuple[int, int, int],
    pool: FastTreeEVPathPool,
    network: NetworkData,
    od: ODData,
    operational_station_indices: np.ndarray,
    current_edge_time: np.ndarray,
) -> List[HybridCandidatePath]:
    ci, oi, dj = (int(value) for value in key)
    source = int(od.ev_node_index[oi])
    sink = int(od.ev_node_index[dj])
    paths: List[HybridCandidatePath] = []
    for slot in range(int(pool.count[ci, oi, dj])):
        kind = int(pool.kind[ci, oi, dj, slot])
        if kind == 0:
            variant = int(pool.direct_tree[ci, oi, dj, slot])
            path = _hybrid_reconstruct_tree_path(
                pool.origin_variant_pred_edge[variant][oi],
                source,
                sink,
                network,
                current_edge_time,
                -1,
            )
            if path is not None:
                paths.append(path)
        elif kind == 1:
            station_local = int(pool.station[ci, oi, dj, slot])
            first_variant = int(pool.first_tree[ci, oi, dj, slot])
            second_variant = int(pool.second_tree[ci, oi, dj, slot])
            if station_local < 0 or station_local >= len(operational_station_indices):
                continue
            station_global = int(operational_station_indices[station_local])
            entrance = int(network.station_entrance_node[station_global])
            exit_node = int(network.station_exit_node[station_global])
            first = _hybrid_reconstruct_tree_path(
                pool.origin_variant_pred_edge[first_variant][oi],
                source,
                entrance,
                network,
                current_edge_time,
                station_local,
            )
            second = _hybrid_reconstruct_tree_path(
                pool.station_variant_pred_edge[second_variant][station_local],
                exit_node,
                sink,
                network,
                current_edge_time,
                station_local,
            )
            if first is None or second is None:
                continue
            sojourn = float(network.station_service_min[station_global])
            combined = _hybrid_combine_station_paths(
                [first], [second], station_local, station_global,
                network, sojourn, 1,
                network.edge_type == "road", 1.0, 0.0,
            )
            paths.extend(combined)
    return _hybrid_merge_unique_paths(paths)


def _hybrid_candidate_record_selection(
    pool: FastTreeEVPathPool,
    od: ODData,
    shares: np.ndarray,
    refined_mask: np.ndarray,
    settings: HybridYenSettings,
    maximum_records: int,
    dynamic: bool,
) -> List[Tuple[Tuple[int, int, int], str]]:
    demand = shares[:, None, None] * od.ev[None, :, :]
    positive = demand > 0.0
    count = pool.count.astype(np.float64)
    available = positive & ~refined_mask.astype(bool)


    if not settings.allow_full_graph_fallback:
        available &= count > 0.0
    if not bool(np.any(available)) or maximum_records <= 0:
        return []

    deficit = np.maximum(0.0, pool.kind.shape[3] - count) / max(1.0, float(pool.kind.shape[3]))
    overlap = 1.0 - pool.mean_pairwise_jaccard_distance.astype(np.float64)
    high_overlap = (count >= 2) & (overlap >= float(settings.overlap_trigger))
    requires = pool.requires_charging.astype(bool)
    station_values = pool.station[pool.station >= 0]
    available_station_count = (int(station_values.max()) + 1) if station_values.size else 0
    station_choice_target = min(
        int(settings.minimum_station_choices),
        max(1, available_station_count),
    )
    low_station_choice = (
        requires
        & (pool.distinct_station_count < station_choice_target)
    )
    priority_violation = (
        pool.priority_station_feasible.astype(bool)
        & ~pool.priority_station_included.astype(bool)
    )
    incomplete = count < float(pool.kind.shape[3])

    critical = available & (
        incomplete | high_overlap | low_station_choice | priority_violation
    )


    flat_available = np.flatnonzero(available.ravel())
    available_demand = demand.ravel()[flat_available]
    order = np.argsort(-available_demand, kind="stable")
    total = float(demand[positive].sum())
    target = max(0.0, min(1.0, float(settings.cumulative_demand_coverage))) * total
    cumulative = 0.0
    top_flat: List[int] = []
    for position in order:
        flat = int(flat_available[int(position)])
        top_flat.append(flat)
        cumulative += float(demand.ravel()[flat])
        if len(top_flat) >= maximum_records or (target > 0.0 and cumulative >= target):
            break

    candidate_flat = set(int(value) for value in np.flatnonzero(critical.ravel()))
    candidate_flat.update(top_flat)
    if not candidate_flat:
        return []

    max_demand = max(1.0e-30, float(demand.ravel()[list(candidate_flat)].max()))
    ranked: List[Tuple[float, float, int, str]] = []
    shape = demand.shape
    for flat in candidate_flat:
        ci, oi, dj = np.unravel_index(flat, shape)
        reasons: List[str] = []
        score = math.log1p(float(demand[ci, oi, dj])) / math.log1p(max_demand)
        if incomplete[ci, oi, dj]:
            reasons.append("path_count_below_K")
            score += 3.0 * float(deficit[ci, oi, dj])
        if high_overlap[ci, oi, dj]:
            reasons.append("high_overlap")
            score += 2.0 * max(0.0, float(overlap[ci, oi, dj]) - settings.overlap_trigger)
        if low_station_choice[ci, oi, dj]:
            reasons.append("low_station_diversity")
            score += 2.0
        if priority_violation[ci, oi, dj]:
            reasons.append("priority_station_missing")
            score += 10.0
        if flat in top_flat:
            reasons.append("high_demand")
            score += 1.0
        if dynamic:
            score += 0.25
            reasons.append("dynamic_enrichment")
        ranked.append((score, float(demand[ci, oi, dj]), flat, ";".join(reasons)))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected: List[Tuple[Tuple[int, int, int], str]] = []
    for _, _, flat, reason in ranked[:maximum_records]:
        ci, oi, dj = np.unravel_index(flat, shape)
        selected.append(((int(ci), int(oi), int(dj)), reason))
    return selected


def _hybrid_generalized_edge_time(
    road_and_bypass_time: np.ndarray,
    qmetrics: QueueMetrics,
    network: NetworkData,
    config: ScenarioConfig,
    operational_station_indices: np.ndarray,
    loading_penalty_min: float,
) -> np.ndarray:
    times = np.asarray(road_and_bypass_time, dtype=np.float64).copy()
    perceived_station_cost = station_choice_costs(qmetrics, network, config)[0]
    for local, global_station in enumerate(operational_station_indices):
        entry = int(network.station_entry_edge[global_station])
        exit_edge = int(network.station_exit_edge[global_station])
        station_cost = float(perceived_station_cost[global_station])
        if not math.isfinite(station_cost):
            station_cost = float(loading_penalty_min)
        times[entry] = station_cost
        times[exit_edge] = 0.0
    return times


def _hybrid_refine_one_record(
    key: Tuple[int, int, int],
    reason: str,
    pool: FastTreeEVPathPool,
    network: NetworkData,
    od: ODData,
    config: ScenarioConfig,
    settings: HybridYenSettings,
    shares: np.ndarray,
    initial_ranges: np.ndarray,
    operational_station_indices: np.ndarray,
    current_travel_time: np.ndarray,
    current_station_sojourn: np.ndarray,
    travel_mask: np.ndarray,
    distance_rank_weight: np.ndarray,
    priority_local: int,
) -> Tuple[Tuple[int, int, int], str, List[HybridCandidatePath]]:
    ci, oi, dj = key
    source = int(od.ev_node_index[oi])
    sink = int(od.ev_node_index[dj])
    r0 = float(initial_ranges[ci])
    road_edge_mask = network.edge_type == "road"
    seed_paths = _hybrid_fast_seed_paths_for_record(
        key, pool, network, od, operational_station_indices, current_travel_time
    )
    direct_possible = not bool(pool.requires_charging[ci, oi, dj])

    if direct_possible:
        direct_seeds = [path for path in seed_paths if path.station_local_index < 0]
        paths = _hybrid_adaptive_corridor_yen(
            seed_paths=direct_seeds,
            source=source,
            sink=sink,
            max_distance_miles=r0,
            network=network,
            travel_mask=travel_mask,
            current_edge_time=current_travel_time,
            distance_rank_weight=distance_rank_weight,
            settings=settings,
            k_target=int(config.ev_initial_k_paths),
            road_edge_mask=road_edge_mask,
            diversity_weight=float(config.ev_path_diversity_weight),
            cost_weight=float(config.ev_path_cost_weight),
        )

        paths = [path for path in paths if path.station_local_index < 0]
        return key, reason, paths[: int(config.ev_initial_k_paths)]

    n_stations = len(operational_station_indices)
    feasible_stations: List[Tuple[float, int]] = []
    for station_local in range(n_stations):
        first_miles = pool.first_variant_miles[:, oi, station_local]
        second_miles = pool.second_variant_miles[:, station_local, dj]
        if not bool(np.any(first_miles <= r0 + 1.0e-9)):
            continue
        if not bool(np.any(second_miles <= config.battery_range_miles + 1.0e-9)):
            continue


        approximate = float(np.nanmin(first_miles) + np.nanmin(second_miles))
        feasible_stations.append((approximate, station_local))

    seed_by_station: Dict[int, List[HybridCandidatePath]] = {}
    for path in seed_paths:
        if path.station_local_index >= 0:
            seed_by_station.setdefault(int(path.station_local_index), []).append(path)

    chosen_stations: List[int] = []
    if priority_local >= 0 and any(station == priority_local for _, station in feasible_stations):
        chosen_stations.append(priority_local)
    for station in sorted(seed_by_station):
        if station not in chosen_stations:
            chosen_stations.append(station)
    for _, station in sorted(feasible_stations):
        if station not in chosen_stations:
            chosen_stations.append(station)
        if len(chosen_stations) >= int(settings.max_stations_per_record):
            break
    chosen_stations = chosen_stations[: max(1, int(settings.max_stations_per_record))]

    candidates_by_station: Dict[int, List[HybridCandidatePath]] = {
        station: list(paths) for station, paths in seed_by_station.items()
    }
    for station_local in chosen_stations:
        station_global = int(operational_station_indices[station_local])
        entrance = int(network.station_entrance_node[station_global])
        exit_node = int(network.station_exit_node[station_global])

        first_seeds: List[HybridCandidatePath] = []
        for variant, predecessor in enumerate(pool.origin_variant_pred_edge):
            if pool.first_variant_miles[variant, oi, station_local] > r0 + 1.0e-9:
                continue
            path = _hybrid_reconstruct_tree_path(
                predecessor[oi], source, entrance, network, current_travel_time,
                station_local,
            )
            if path is not None and path.distance_miles <= r0 + 1.0e-9:
                first_seeds.append(path)
        second_seeds: List[HybridCandidatePath] = []
        for variant, predecessor in enumerate(pool.station_variant_pred_edge):
            if (
                pool.second_variant_miles[variant, station_local, dj]
                > config.battery_range_miles + 1.0e-9
            ):
                continue
            path = _hybrid_reconstruct_tree_path(
                predecessor[station_local], exit_node, sink, network,
                current_travel_time, station_local,
            )
            if path is not None and path.distance_miles <= config.battery_range_miles + 1.0e-9:
                second_seeds.append(path)

        first_paths = _hybrid_adaptive_corridor_yen(
            seed_paths=_hybrid_merge_unique_paths(first_seeds),
            source=source,
            sink=entrance,
            max_distance_miles=r0,
            network=network,
            travel_mask=travel_mask,
            current_edge_time=current_travel_time,
            distance_rank_weight=distance_rank_weight,
            settings=settings,
            k_target=min(settings.search_k, config.ev_initial_k_paths),
            road_edge_mask=road_edge_mask,
            diversity_weight=float(config.ev_path_diversity_weight),
            cost_weight=float(config.ev_path_cost_weight),
        )
        second_paths = _hybrid_adaptive_corridor_yen(
            seed_paths=_hybrid_merge_unique_paths(second_seeds),
            source=exit_node,
            sink=sink,
            max_distance_miles=float(config.battery_range_miles),
            network=network,
            travel_mask=travel_mask,
            current_edge_time=current_travel_time,
            distance_rank_weight=distance_rank_weight,
            settings=settings,
            k_target=min(settings.search_k, config.ev_initial_k_paths),
            road_edge_mask=road_edge_mask,
            diversity_weight=float(config.ev_path_diversity_weight),
            cost_weight=float(config.ev_path_cost_weight),
        )
        if not first_paths or not second_paths:
            continue
        station_sojourn = float(current_station_sojourn[station_local])
        if not math.isfinite(station_sojourn):
            station_sojourn = float(config.queue_unstable_loading_penalty_min)
        combined = _hybrid_combine_station_paths(
            first_paths,
            second_paths,
            station_local,
            station_global,
            network,
            station_sojourn,
            int(config.ev_initial_k_paths),
            road_edge_mask,
            float(config.ev_path_diversity_weight),
            float(config.ev_path_cost_weight),
        )
        candidates_by_station[station_local] = _hybrid_merge_unique_paths(
            candidates_by_station.get(station_local, []), combined
        )

    selected = _hybrid_select_station_diverse(
        candidates_by_station,
        int(config.ev_initial_k_paths),
        road_edge_mask,
        float(config.ev_path_diversity_weight),
        float(config.ev_path_cost_weight),
        int(priority_local),
        bool(config.ev_require_priority_station_if_feasible),
    )
    return key, reason, selected


def _hybrid_compact_pool(
    records: Mapping[Tuple[int, int, int], Sequence[HybridCandidatePath]],
    reasons: Mapping[Tuple[int, int, int], str],
    rounds: Mapping[Tuple[int, int, int], int],
    od: ODData,
    shares: np.ndarray,
) -> HybridExplicitPathPool:
    keys = sorted(records)
    record_class: List[int] = []
    record_origin: List[int] = []
    record_dest: List[int] = []
    record_demand: List[float] = []
    record_offsets: List[int] = [0]
    path_station: List[int] = []
    edge_offsets: List[int] = [0]
    edges: List[int] = []
    record_round: List[int] = []
    record_reason: List[str] = []
    for ci, oi, dj in keys:
        paths = list(records[(ci, oi, dj)])
        record_class.append(int(ci))
        record_origin.append(int(oi))
        record_dest.append(int(dj))
        record_demand.append(float(od.ev[oi, dj] * shares[ci]))
        record_round.append(int(rounds.get((ci, oi, dj), 0)))
        record_reason.append(str(reasons.get((ci, oi, dj), "")))
        for path in paths:
            path_station.append(int(path.station_local_index))
            edges.extend(int(edge) for edge in path.edges)
            edge_offsets.append(len(edges))
        record_offsets.append(len(path_station))
    return HybridExplicitPathPool(
        record_class=np.asarray(record_class, dtype=np.int8),
        record_origin=np.asarray(record_origin, dtype=np.int16),
        record_dest=np.asarray(record_dest, dtype=np.int16),
        record_demand=np.asarray(record_demand, dtype=np.float64),
        record_path_offsets=np.asarray(record_offsets, dtype=np.int64),
        path_station=np.asarray(path_station, dtype=np.int16),
        path_edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
        path_edges=np.asarray(edges, dtype=np.int32),
        record_round=np.asarray(record_round, dtype=np.int16),
        record_reason=np.asarray(record_reason, dtype="U160"),
    )


def _hybrid_mask_from_compact(
    compact: HybridExplicitPathPool,
    shape: Tuple[int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if compact.n_records:
        mask[
            compact.record_class.astype(np.int64),
            compact.record_origin.astype(np.int64),
            compact.record_dest.astype(np.int64),
        ] = 1
    return mask


def _hybrid_records_from_compact(
    compact: HybridExplicitPathPool,
    network: NetworkData,
    edge_time: np.ndarray,
) -> Tuple[
    Dict[Tuple[int, int, int], List[HybridCandidatePath]],
    Dict[Tuple[int, int, int], str],
    Dict[Tuple[int, int, int], int],
]:
    records: Dict[Tuple[int, int, int], List[HybridCandidatePath]] = {}
    reasons: Dict[Tuple[int, int, int], str] = {}
    rounds: Dict[Tuple[int, int, int], int] = {}
    for rr in range(compact.n_records):
        key = (
            int(compact.record_class[rr]),
            int(compact.record_origin[rr]),
            int(compact.record_dest[rr]),
        )
        paths: List[HybridCandidatePath] = []
        for pp in range(
            int(compact.record_path_offsets[rr]),
            int(compact.record_path_offsets[rr + 1]),
        ):
            e0 = int(compact.path_edge_offsets[pp])
            e1 = int(compact.path_edge_offsets[pp + 1])
            path_edges = compact.path_edges[e0:e1].astype(np.int32, copy=True)
            if path_edges.size:
                nodes = np.concatenate([
                    np.asarray([network.edge_u[int(path_edges[0])]], dtype=np.int32),
                    network.edge_v[path_edges].astype(np.int32),
                ])
            else:
                nodes = np.empty(0, dtype=np.int32)
            paths.append(HybridCandidatePath(
                nodes=nodes,
                edges=path_edges,
                base_cost_min=float(edge_time[path_edges].sum()) if path_edges.size else 0.0,
                distance_miles=float(network.edge_length_miles[path_edges].sum()) if path_edges.size else 0.0,
                station_local_index=int(compact.path_station[pp]),
            ))
        records[key] = paths
        reasons[key] = str(compact.record_reason[rr])
        rounds[key] = int(compact.record_round[rr])
    return records, reasons, rounds


def _hybrid_cache_fingerprint(
    network: NetworkData,
    od: ODData,
    config: ScenarioConfig,
    settings: HybridYenSettings,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(config.scenario_id).encode("utf-8"))
    digest.update(network.edge_u.astype(np.int32).tobytes())
    digest.update(network.edge_v.astype(np.int32).tobytes())
    digest.update(network.active_edge.astype(np.uint8).tobytes())
    selected_t0 = (
        network.edge_t0_flood_min if config.use_flood_road_state
        else network.edge_t0_dry_min
    )
    digest.update(np.nan_to_num(selected_t0, posinf=1.0e30).astype(np.float64).tobytes())
    digest.update(od.ev.astype(np.float64).tobytes())
    digest.update(json.dumps({
        "K": int(config.ev_initial_k_paths),
        "tree_variants": int(config.ev_direct_candidate_trees),
        "priority": str(config.ev_priority_station_id),
        "queue_wait_disutility_multiplier": float(config.queue_wait_disutility_multiplier),
        "queue_utilization_penalty_start": float(config.queue_utilization_penalty_start),
        "queue_utilization_penalty_scale_min": float(config.queue_utilization_penalty_scale_min),
        "queue_utilization_penalty_power": float(config.queue_utilization_penalty_power),
        "queue_utilization_penalty_cap_min": float(config.queue_utilization_penalty_cap_min),
        "settings": settings.__dict__,
    }, sort_keys=True, default=list).encode("utf-8"))
    return digest.hexdigest()


def _hybrid_save_compact(path: Path, compact: HybridExplicitPathPool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        record_class=compact.record_class,
        record_origin=compact.record_origin,
        record_dest=compact.record_dest,
        record_demand=compact.record_demand,
        record_path_offsets=compact.record_path_offsets,
        path_station=compact.path_station,
        path_edge_offsets=compact.path_edge_offsets,
        path_edges=compact.path_edges,
        record_round=compact.record_round,
        record_reason=compact.record_reason,
    )
    temporary.replace(path)


def _hybrid_load_compact(path: Path) -> HybridExplicitPathPool:
    with np.load(path, allow_pickle=False) as data:
        return HybridExplicitPathPool(
            record_class=data["record_class"].astype(np.int8),
            record_origin=data["record_origin"].astype(np.int16),
            record_dest=data["record_dest"].astype(np.int16),
            record_demand=data["record_demand"].astype(np.float64),
            record_path_offsets=data["record_path_offsets"].astype(np.int64),
            path_station=data["path_station"].astype(np.int16),
            path_edge_offsets=data["path_edge_offsets"].astype(np.int64),
            path_edges=data["path_edges"].astype(np.int32),
            record_round=data["record_round"].astype(np.int16),
            record_reason=data["record_reason"].astype("U160"),
        )


def _hybrid_refine_selected_records(
    selected: Sequence[Tuple[Tuple[int, int, int], str]],
    state: HybridRefinementState,
    refinement_round: int,
    pool: FastTreeEVPathPool,
    network: NetworkData,
    od: ODData,
    config: ScenarioConfig,
    settings: HybridYenSettings,
    shares: np.ndarray,
    initial_ranges: np.ndarray,
    operational_station_indices: np.ndarray,
    current_travel_time: np.ndarray,
    current_station_sojourn: np.ndarray,
    travel_mask: np.ndarray,
    distance_rank_weight: np.ndarray,
    priority_local: int,
) -> int:
    if not selected:
        return 0
    started = time.perf_counter()
    completed = 0

    def worker(item: Tuple[Tuple[int, int, int], str]):
        key, reason = item
        return _hybrid_refine_one_record(
            key, reason, pool, network, od, config, settings,
            shares, initial_ranges, operational_station_indices,
            current_travel_time, current_station_sojourn,
            travel_mask, distance_rank_weight, priority_local,
        )

    n_jobs = max(1, int(settings.n_jobs))
    if n_jobs == 1:
        results = [worker(item) for item in selected]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(worker, item): item for item in selected}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    key, _ = futures[future]
                    print(f"  Hybrid Yen warning for record {key}: {exc}")
                if index == 1 or index % max(1, settings.progress_every) == 0:
                    print(
                        f"  Hybrid Yen refinement: completed {index:,}/{len(selected):,} records."
                    )

    for key, reason, paths in results:
        if not paths:
            continue
        state.records[key] = list(paths[: int(config.ev_initial_k_paths)])
        state.reasons[key] = reason
        state.rounds[key] = int(refinement_round)
        completed += 1

    state.compact = _hybrid_compact_pool(
        state.records, state.reasons, state.rounds, od, shares
    )
    state.mask = _hybrid_mask_from_compact(
        state.compact,
        (len(shares), od.ev.shape[0], od.ev.shape[1]),
    )
    elapsed = time.perf_counter() - started
    if refinement_round == 0:
        state.initial_build_seconds += elapsed
    else:
        state.dynamic_build_seconds += elapsed
        state.enrichment_rounds += 1
    print(
        f"Hybrid Yen round {refinement_round}: retained {completed:,}/{len(selected):,} "
        f"new refined records in {elapsed:.2f}s; total refined={state.compact.n_records:,}."
    )
    return completed


@njit(cache=False)
def hybrid_explicit_logit_loading(
    record_class: np.ndarray,
    record_demand: np.ndarray,
    record_path_offsets: np.ndarray,
    path_station: np.ndarray,
    path_edge_offsets: np.ndarray,
    path_edges: np.ndarray,
    generalized_edge_time: np.ndarray,
    theta: float,
    n_edges: int,
    n_stations: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_flow = np.zeros(n_edges, dtype=np.float64)
    station_entries = np.zeros(n_stations, dtype=np.float64)
    served_by_class = np.zeros(n_classes, dtype=np.float64)
    potential_by_class = np.zeros(n_classes, dtype=np.float64)
    unserved_by_class = np.zeros(n_classes, dtype=np.float64)
    costs = np.empty(16, dtype=np.float64)
    weights = np.empty(16, dtype=np.float64)
    for rr in range(record_demand.size):
        q = float(record_demand[rr])
        ci = int(record_class[rr])
        if q <= 0.0:
            continue
        potential_by_class[ci] += q
        start = int(record_path_offsets[rr])
        stop = int(record_path_offsets[rr + 1])
        count = min(stop - start, 16)
        if count <= 0:
            unserved_by_class[ci] += q
            continue
        cmin = np.inf
        for local in range(count):
            path = start + local
            total = 0.0
            for position in range(
                int(path_edge_offsets[path]),
                int(path_edge_offsets[path + 1]),
            ):
                total += generalized_edge_time[int(path_edges[position])]
            costs[local] = total
            if total < cmin:
                cmin = total
        denom = 0.0
        for local in range(count):
            exponent = -theta * (costs[local] - cmin)
            weight = 0.0 if exponent < -745.0 else math.exp(exponent)
            weights[local] = weight
            denom += weight
        if denom <= 0.0 or not math.isfinite(denom):
            best = 0
            for local in range(1, count):
                if costs[local] < costs[best]:
                    best = local
            for local in range(count):
                weights[local] = 1.0 if local == best else 0.0
            denom = 1.0
        served_by_class[ci] += q
        for local in range(count):
            if weights[local] <= 0.0:
                continue
            path = start + local
            flow = q * weights[local] / denom
            for position in range(
                int(path_edge_offsets[path]),
                int(path_edge_offsets[path + 1]),
            ):
                edge_flow[int(path_edges[position])] += flow
            station = int(path_station[path])
            if station >= 0:
                station_entries[station] += flow
    return edge_flow, station_entries, served_by_class, potential_by_class, unserved_by_class


def _hybrid_refinement_summary(
    state: HybridRefinementState,
    od: ODData,
) -> pd.DataFrame:
    compact = state.compact
    total_demand = float(od.potential_ev)
    refined_demand = float(compact.record_demand.sum())
    counts = np.diff(compact.record_path_offsets) if compact.n_records else np.empty(0)
    return pd.DataFrame([{
        "refined_records": compact.n_records,
        "refined_paths": compact.n_paths,
        "refined_path_edge_entries": compact.n_path_edges,
        "refined_ev_demand": refined_demand,
        "refined_ev_demand_share": refined_demand / total_demand if total_demand > 0 else float("nan"),
        "average_paths_per_refined_record": float(counts.mean()) if counts.size else 0.0,
        "minimum_paths_per_refined_record": int(counts.min()) if counts.size else 0,
        "maximum_paths_per_refined_record": int(counts.max()) if counts.size else 0,
        "initial_refinement_seconds": state.initial_build_seconds,
        "dynamic_refinement_seconds": state.dynamic_build_seconds,
        "dynamic_enrichment_rounds": state.enrichment_rounds,
    }])


def _hybrid_refined_record_table(
    state: HybridRefinementState,
    od: ODData,
    config: ScenarioConfig,
) -> pd.DataFrame:
    compact = state.compact
    rows: List[Dict[str, object]] = []
    counts = np.diff(compact.record_path_offsets)
    for rr in range(compact.n_records):
        ci = int(compact.record_class[rr])
        oi = int(compact.record_origin[rr])
        dj = int(compact.record_dest[rr])
        stations = {
            int(compact.path_station[pp])
            for pp in range(
                int(compact.record_path_offsets[rr]),
                int(compact.record_path_offsets[rr + 1]),
            )
            if int(compact.path_station[pp]) >= 0
        }
        rows.append({
            "soc_fraction": float(config.initial_soc_fractions[ci]),
            "origin": int(od.ev_node_ids[oi]),
            "destination": int(od.ev_node_ids[dj]),
            "demand": float(compact.record_demand[rr]),
            "retained_paths": int(counts[rr]),
            "distinct_charging_stations": len(stations),
            "refinement_round": int(compact.record_round[rr]),
            "selection_reason": str(compact.record_reason[rr]),
        })
    return pd.DataFrame(rows)


_fast_base_write_outputs = write_outputs


def perform_assignment(
    network: NetworkData,
    od: ODData,
    config: ScenarioConfig,
    path_cache_root: Optional[Path] = None,
) -> AssignmentResult:

    settings = HYBRID_YEN_SETTINGS
    validate_queue_choice_settings(config)
    if len(config.initial_soc_fractions) != len(config.ev_class_shares):
        raise ValueError("initial_soc_fractions and ev_class_shares must have equal lengths.")
    if config.ev_initial_k_paths <= 0:
        raise ValueError("ev_initial_k_paths must be positive.")
    if config.ev_direct_candidate_trees < config.ev_initial_k_paths:
        raise ValueError("EV_DIRECT_CANDIDATE_TREES must be at least K.")

    shares = np.asarray(config.ev_class_shares, dtype=np.float64)
    shares /= shares.sum()
    initial_ranges = config.battery_range_miles * np.asarray(
        config.initial_soc_fractions, dtype=np.float64
    )
    n_edges = len(network.links)
    n_nodes = len(network.node_ids)
    operational_station_indices = np.where(network.operational_station_mask)[0]
    op_station_ids = [network.station_ids[index] for index in operational_station_indices]
    op_entrance_nodes = network.station_entrance_node[operational_station_indices]
    op_exit_nodes = network.station_exit_node[operational_station_indices]
    n_ev_orig = len(od.ev_node_index)
    dest = od.ev_node_index

    priority_local = -1
    priority_id = str(config.ev_priority_station_id).strip().upper()
    for local, station_id in enumerate(op_station_ids):
        if str(station_id).upper() == priority_id:
            priority_local = local
            break

    travel_mask = network.travel_edge.copy()
    _, out_indptr, out_edges, in_indptr, in_edges = build_adjacency(
        network.edge_u, network.edge_v, travel_mask
    )
    road_mask = network.edge_type == "road"
    bypass_mask = network.edge_type == "byp"
    distance_rank_weight = np.full(n_edges, INF, dtype=np.float64)
    distance_rank_weight[road_mask & travel_mask] = np.maximum(
        network.edge_length_miles[road_mask & travel_mask],
        ALGORITHM_EPS_DISTANCE_MILE,
    )
    distance_rank_weight[bypass_mask & travel_mask] = ALGORITHM_EPS_DISTANCE_MILE

    ev_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)
    distance_sparse, distance_pair_keys, distance_pair_edges = min_pair_sparse(
        n_nodes, network.edge_u, network.edge_v,
        distance_rank_weight, travel_mask,
    )
    distance_matrix, distance_pred_node = dijkstra(
        distance_sparse, directed=True, indices=ev_sources,
        return_predecessors=True,
    )
    distance_pred_edge = predecessor_nodes_to_edges(
        distance_pred_node, distance_pair_keys, distance_pair_edges, n_nodes
    ).astype(np.int32, copy=False)
    distance_orig = distance_matrix[:n_ev_orig]
    distance_station = distance_matrix[n_ev_orig:]
    distance_pred_orig = distance_pred_edge[:n_ev_orig]
    distance_pred_station = distance_pred_edge[n_ev_orig:]
    distance_order_orig = np.argsort(distance_orig, axis=1).astype(np.int32)
    distance_order_station = np.argsort(distance_station, axis=1).astype(np.int32)
    direct_shortest_miles = distance_orig[:, dest]

    edge_minhash = build_edge_minhash(
        network.edge_type,
        config.ev_path_minhash_size,
        config.ev_path_random_seed,
    )

    non_flow = np.zeros(n_edges, dtype=np.float64)
    ev_flow = np.zeros(n_edges, dtype=np.float64)
    station_entries_full = np.zeros(len(network.station_ids), dtype=np.float64)
    served_ev_by_class = np.zeros(len(shares), dtype=np.float64)
    potential_ev_by_class = od.potential_ev * shares
    served_non_ev = 0.0
    converged = False
    gap = INF
    fast_pool: Optional[FastTreeEVPathPool] = None
    fast_pool_summary: Optional[pd.DataFrame] = None
    pool_refreshes = 0

    empty_compact = HybridExplicitPathPool(
        record_class=np.empty(0, dtype=np.int8),
        record_origin=np.empty(0, dtype=np.int16),
        record_dest=np.empty(0, dtype=np.int16),
        record_demand=np.empty(0, dtype=np.float64),
        record_path_offsets=np.asarray([0], dtype=np.int64),
        path_station=np.empty(0, dtype=np.int16),
        path_edge_offsets=np.asarray([0], dtype=np.int64),
        path_edges=np.empty(0, dtype=np.int32),
        record_round=np.empty(0, dtype=np.int16),
        record_reason=np.empty(0, dtype="U160"),
    )
    refinement_state = HybridRefinementState(
        records={}, reasons={}, rounds={}, compact=empty_compact,
        mask=np.zeros((len(shares), od.ev.shape[0], od.ev.shape[1]), dtype=np.uint8),
    )

    non_ev_sources = od.od_node_index.astype(np.int64)
    ev_time_sources = np.concatenate([od.ev_node_index, op_exit_nodes]).astype(np.int64)
    cache_root = Path(path_cache_root) if path_cache_root is not None else Path.cwd() / "hybrid_yen_cache"

    print(f"[{config.scenario_label}] loaded {len(network.links):,} links, {len(network.node_ids):,} nodes")
    print(f"[{config.scenario_label}] OD nodes={len(od.od_node_ids):,}; EV representative nodes={len(od.ev_node_ids):,}")
    print(f"[{config.scenario_label}] operational stations={op_station_ids}")
    print(f"[{config.scenario_label}] potential non-EV={od.potential_non_ev:.8f}; potential EV={od.potential_ev:.8f}")
    print(
        f"[{config.scenario_label}] HYBRID EV paths: all records use {config.ev_direct_candidate_trees} "
        f"batched tree variants; up to {settings.initial_max_records:,} critical records "
        f"receive adaptive-corridor Yen refinement; loading K={config.ev_initial_k_paths}."
    )

    stage_timing = os.environ.get("CHICAGO_STAGE_TIMING", "0") == "1"
    for iteration in range(1, config.max_msa_iterations + 1):
        iteration_clock = time.perf_counter()
        old_non = non_flow.copy()
        old_ev = ev_flow.copy()
        old_entries = station_entries_full.copy()
        total_old = old_non + old_ev
        link_time = road_times(total_old, network, config)
        qmetrics = queue_metrics(old_entries, network, config)

        algorithm_time = np.full(n_edges, INF, dtype=np.float64)
        algorithm_time[travel_mask] = np.maximum(
            link_time[travel_mask], ALGORITHM_EPS_TIME_MIN
        )
        time_sparse, time_pair_keys, time_pair_edges = min_pair_sparse(
            n_nodes, network.edge_u, network.edge_v,
            algorithm_time, travel_mask,
        )

        non_dist = dijkstra(
            time_sparse, directed=True, indices=non_ev_sources,
            return_predecessors=False,
        )
        non_aux, non_unassigned = dial_logit_load_all_origins(
            non_dist, od.non_ev, od.od_node_index, od.od_node_index,
            network.edge_u, network.edge_v, algorithm_time,
            out_indptr, out_edges, in_indptr, in_edges,
            config.theta_non_ev,
        )
        if iteration == 1:
            served_non_ev = od.potential_non_ev - float(non_unassigned)
        del non_dist

        ev_all_dist, ev_all_pred_node = dijkstra(
            time_sparse, directed=True, indices=ev_time_sources,
            return_predecessors=True,
        )
        ev_pred_node = ev_all_pred_node[:n_ev_orig]
        station_pred_node = ev_all_pred_node[n_ev_orig:]
        ev_pred_edge = predecessor_nodes_to_edges(
            ev_pred_node, time_pair_keys, time_pair_edges, n_nodes
        ).astype(np.int32, copy=False)
        station_pred_edge = predecessor_nodes_to_edges(
            station_pred_node, time_pair_keys, time_pair_edges, n_nodes
        ).astype(np.int32, copy=False)
        ev_time_dist = ev_all_dist[:n_ev_orig]
        station_time_dist = ev_all_dist[n_ev_orig:]
        time_order_orig = np.argsort(ev_time_dist, axis=1).astype(np.int32)
        time_order_station = np.argsort(station_time_dist, axis=1).astype(np.int32)

        refresh_fast_pool = (
            fast_pool is None
            or (
                config.ev_path_refresh_every > 0
                and iteration % config.ev_path_refresh_every == 0
            )
        )
        if refresh_fast_pool:
            pool_clock = time.perf_counter()
            origin_targets = np.concatenate([dest, op_entrance_nodes]).astype(np.int64)
            (
                origin_preds, origin_orders, origin_cost_all,
                origin_miles_all, origin_sig_all,
            ) = build_direct_route_tree_variants(
                n_nodes=n_nodes,
                edge_u=network.edge_u,
                edge_v=network.edge_v,
                edge_type=network.edge_type,
                travel_mask=travel_mask,
                algorithm_time=algorithm_time,
                source_nodes=od.ev_node_index,
                destination_nodes=origin_targets,
                edge_length_miles=network.edge_length_miles,
                edge_minhash=edge_minhash,
                base_time_pred_edge=ev_pred_edge,
                base_time_order=time_order_orig,
                distance_pred_edge=distance_pred_orig,
                distance_order=distance_order_orig,
                candidate_count=config.ev_direct_candidate_trees,
                perturbation_strength=config.ev_direct_perturbation_strength,
                random_seed=config.ev_path_random_seed + iteration,
            )
            (
                station_preds, station_orders, second_cost,
                second_miles, second_sig,
            ) = build_direct_route_tree_variants(
                n_nodes=n_nodes,
                edge_u=network.edge_u,
                edge_v=network.edge_v,
                edge_type=network.edge_type,
                travel_mask=travel_mask,
                algorithm_time=algorithm_time,
                source_nodes=op_exit_nodes,
                destination_nodes=dest,
                edge_length_miles=network.edge_length_miles,
                edge_minhash=edge_minhash,
                base_time_pred_edge=station_pred_edge,
                base_time_order=time_order_station,
                distance_pred_edge=distance_pred_station,
                distance_order=distance_order_station,
                candidate_count=config.ev_direct_candidate_trees,
                perturbation_strength=config.ev_direct_perturbation_strength,
                random_seed=config.ev_path_random_seed + 100000 + iteration,
            )
            n_dest = len(dest)
            direct_cost = origin_cost_all[:, :, :n_dest]
            direct_miles = origin_miles_all[:, :, :n_dest]
            direct_sig = origin_sig_all[:, :, :n_dest, :]
            first_cost = origin_cost_all[:, :, n_dest:]
            first_miles = origin_miles_all[:, :, n_dest:]
            first_sig = origin_sig_all[:, :, n_dest:, :]
            station_sojourn = station_choice_costs(qmetrics, network, config)[0][operational_station_indices]

            pair_first: List[int] = []
            pair_second: List[int] = []
            for variant in range(min(direct_cost.shape[0], second_cost.shape[0])):
                pair_first.append(variant); pair_second.append(variant)
            for variant in range(1, direct_cost.shape[0]):
                pair_first.append(variant); pair_second.append(0)
            for variant in range(1, second_cost.shape[0]):
                pair_first.append(0); pair_second.append(variant)

            (
                pool_kind, pool_station, pool_direct_tree, pool_first_tree,
                pool_second_tree, pool_count, pool_distinct_stations,
                pool_mean_jaccard, pool_requires_charging,
                pool_priority_feasible, pool_priority_included,
            ) = build_fast_tree_ev_path_pool(
                od.ev, shares, initial_ranges, config.battery_range_miles,
                direct_cost, direct_miles, direct_shortest_miles,
                first_cost, first_miles, second_cost, second_miles,
                station_sojourn, direct_sig, first_sig, second_sig,
                config.ev_initial_k_paths,
                config.ev_prefer_distinct_stations,
                config.ev_path_diversity_weight,
                config.ev_path_cost_weight,
                config.queue_unstable_loading_penalty_min,
                int(priority_local),
                bool(config.ev_require_priority_station_if_feasible),
                np.asarray(pair_first, dtype=np.int8),
                np.asarray(pair_second, dtype=np.int8),
            )
            fast_pool = FastTreeEVPathPool(
                kind=pool_kind,
                station=pool_station,
                direct_tree=pool_direct_tree,
                first_tree=pool_first_tree,
                second_tree=pool_second_tree,
                count=pool_count,
                distinct_station_count=pool_distinct_stations,
                mean_pairwise_jaccard_distance=pool_mean_jaccard,
                requires_charging=pool_requires_charging,
                priority_station_feasible=pool_priority_feasible,
                priority_station_included=pool_priority_included,
                origin_variant_pred_edge=origin_preds,
                origin_variant_order=origin_orders,
                station_variant_pred_edge=station_preds,
                station_variant_order=station_orders,
                direct_variant_miles=direct_miles,
                first_variant_miles=first_miles,
                second_variant_miles=second_miles,
                refresh_iteration=iteration,
            )
            fast_pool_summary = _append_priority_metrics(
                summarize_ev_path_pool(
                    fast_pool, od.ev, shares,
                    config.initial_soc_fractions,
                    config.battery_range_miles,
                    config.ev_initial_k_paths,
                ),
                fast_pool, od.ev, shares,
            )
            pool_refreshes += 1
            print(
                f"[{config.scenario_label}] fast tree pool built at MSA {iteration} "
                f"in {time.perf_counter() - pool_clock:.2f}s."
            )
            del origin_cost_all, origin_miles_all, origin_sig_all
            del direct_cost, first_cost, second_cost, direct_sig, first_sig, second_sig

        if fast_pool is None:
            raise RuntimeError("Fast tree path pool was not initialized.")


        if settings.enabled and iteration == 1 and refinement_state.compact.n_records == 0:
            fingerprint = _hybrid_cache_fingerprint(network, od, config, settings)
            cache_file = cache_root / fingerprint[:16] / "initial_hybrid_yen_pool.npz"
            if (
                settings.cache_initial_pool
                and cache_file.exists()
                and not settings.rebuild_cache
            ):
                refinement_state.compact = _hybrid_load_compact(cache_file)
                (
                    refinement_state.records,
                    refinement_state.reasons,
                    refinement_state.rounds,
                ) = _hybrid_records_from_compact(
                    refinement_state.compact, network, algorithm_time
                )
                refinement_state.mask = _hybrid_mask_from_compact(
                    refinement_state.compact,
                    (len(shares), od.ev.shape[0], od.ev.shape[1]),
                )
                print(
                    f"Loaded cached hybrid Yen refinement: "
                    f"{refinement_state.compact.n_records:,} records, "
                    f"{refinement_state.compact.n_paths:,} paths."
                )
            else:
                selected = _hybrid_candidate_record_selection(
                    fast_pool, od, shares, refinement_state.mask,
                    settings, settings.initial_max_records, False,
                )
                print(
                    f"Initial selective Yen refinement: {len(selected):,} critical "
                    f"records, threads={max(1, settings.n_jobs)}."
                )
                _hybrid_refine_selected_records(
                    selected, refinement_state, 0, fast_pool,
                    network, od, config, settings, shares, initial_ranges,
                    operational_station_indices, algorithm_time,
                    station_choice_costs(qmetrics, network, config)[0][operational_station_indices],
                    travel_mask, distance_rank_weight, priority_local,
                )
                if settings.cache_initial_pool and refinement_state.compact.n_records:
                    _hybrid_save_compact(cache_file, refinement_state.compact)


        if (
            settings.enabled
            and settings.enrich_every > 0
            and iteration > 1
            and iteration % settings.enrich_every == 0
            and settings.enrich_top_m > 0
        ):
            selected = _hybrid_candidate_record_selection(
                fast_pool, od, shares, refinement_state.mask,
                settings, settings.enrich_top_m, True,
            )
            if selected:
                _hybrid_refine_selected_records(
                    selected, refinement_state, iteration, fast_pool,
                    network, od, config, settings, shares, initial_ranges,
                    operational_station_indices, algorithm_time,
                    station_choice_costs(qmetrics, network, config)[0][operational_station_indices],
                    travel_mask, distance_rank_weight, priority_local,
                )

        origin_targets = np.concatenate([dest, op_entrance_nodes]).astype(np.int64)
        origin_current = _current_variant_target_costs(
            fast_pool.origin_variant_pred_edge,
            fast_pool.origin_variant_order,
            od.ev_node_index,
            origin_targets,
            network.edge_u,
            algorithm_time,
        )
        station_current = _current_variant_target_costs(
            fast_pool.station_variant_pred_edge,
            fast_pool.station_variant_order,
            op_exit_nodes,
            dest,
            network.edge_u,
            algorithm_time,
        )
        n_dest = len(dest)
        direct_current = origin_current[:, :, :n_dest]
        first_current = origin_current[:, :, n_dest:]
        station_sojourn = station_choice_costs(qmetrics, network, config)[0][operational_station_indices]


        loading_count = fast_pool.count.copy()
        if refinement_state.compact.n_records:
            loading_count[refinement_state.mask.astype(bool)] = 0

        (
            direct_q, first_q, second_q, tree_entries,
            tree_served, _tree_potential, _tree_unserved,
        ) = fast_tree_ev_logit_loading(
            od.ev, shares, initial_ranges, config.battery_range_miles,
            config.theta_ev,
            fast_pool.kind, fast_pool.station,
            fast_pool.direct_tree, fast_pool.first_tree,
            fast_pool.second_tree, loading_count,
            direct_current, fast_pool.direct_variant_miles,
            first_current, fast_pool.first_variant_miles,
            station_current, fast_pool.second_variant_miles,
            station_sojourn, config.queue_unstable_loading_penalty_min,
        )

        ev_aux = np.zeros(n_edges, dtype=np.float64)
        for variant in range(direct_q.shape[0]):
            ev_aux += load_origin_tree_flows(
                fast_pool.origin_variant_pred_edge[variant],
                fast_pool.origin_variant_order[variant],
                direct_q[variant], dest,
                first_q[variant], op_entrance_nodes,
                network.edge_u, n_edges,
            )
        for variant in range(second_q.shape[0]):
            ev_aux += load_station_tree_flows(
                fast_pool.station_variant_pred_edge[variant],
                fast_pool.station_variant_order[variant],
                second_q[variant], dest,
                network.edge_u, n_edges,
            )

        explicit_entries = np.zeros(len(operational_station_indices), dtype=np.float64)
        explicit_served = np.zeros(len(shares), dtype=np.float64)
        if refinement_state.compact.n_records:
            generalized_time = _hybrid_generalized_edge_time(
                link_time, qmetrics, network, config, operational_station_indices, config.queue_unstable_loading_penalty_min
            )
            (
                explicit_flow, explicit_entries, explicit_served,
                _explicit_potential, _explicit_unserved,
            ) = hybrid_explicit_logit_loading(
                refinement_state.compact.record_class,
                refinement_state.compact.record_demand,
                refinement_state.compact.record_path_offsets,
                refinement_state.compact.path_station,
                refinement_state.compact.path_edge_offsets,
                refinement_state.compact.path_edges,
                generalized_time,
                config.theta_ev,
                n_edges,
                len(operational_station_indices),
                len(shares),
            )
            ev_aux += explicit_flow
            del explicit_flow, generalized_time

        entries_op = tree_entries + explicit_entries
        entries_aux_full = np.zeros(len(network.station_ids), dtype=np.float64)
        entries_aux_full[operational_station_indices] = entries_op
        for local, global_station in enumerate(operational_station_indices):
            ev_aux[int(network.station_entry_edge[global_station])] = entries_op[local]
            ev_aux[int(network.station_exit_edge[global_station])] = entries_op[local]

        served_by_class = tree_served + explicit_served
        potential_by_class_iter = od.potential_ev * shares

        step = 1.0 / float(iteration)
        non_flow = old_non + step * (non_aux - old_non)
        ev_flow = old_ev + step * (ev_aux - old_ev)
        station_entries_full = old_entries + step * (entries_aux_full - old_entries)
        served_ev_by_class = served_by_class
        potential_ev_by_class = potential_by_class_iter

        numerator = (
            np.abs(non_flow - old_non).sum()
            + np.abs(ev_flow - old_ev).sum()
            + np.abs(station_entries_full - old_entries).sum()
        )
        denominator = max(
            1.0,
            np.abs(old_non).sum() + np.abs(old_ev).sum() + np.abs(old_entries).sum(),
        )
        gap = float(numerator / denominator)

        del non_aux, ev_aux, time_sparse, ev_all_dist, ev_all_pred_node
        del ev_pred_node, station_pred_node, ev_pred_edge, station_pred_edge
        del ev_time_dist, station_time_dist, time_order_orig, time_order_station
        del origin_current, station_current, direct_current, first_current
        del direct_q, first_q, second_q, loading_count
        gc.collect()

        if stage_timing:
            print(
                f"  hybrid iter {iteration} total: "
                f"{time.perf_counter() - iteration_clock:.3f}s",
                flush=True,
            )
        if iteration == 1 or iteration % max(1, config.progress_every) == 0:
            metrics_now = queue_metrics(station_entries_full, network, config)
            compact_text = ", ".join(
                f"{network.station_ids[index]}:entry={station_entries_full[index]:.2f},"
                f"u={metrics_now.utilization[index]:.3f}"
                for index in operational_station_indices
            )
            print(
                f"[{config.scenario_label}] MSA {iteration}: gap={gap:.6e}; "
                f"EV feasible={served_ev_by_class.sum():.6f}; "
                f"Yen-refined records={refinement_state.compact.n_records:,}; "
                f"{compact_text}"
            )
        if iteration >= config.min_msa_iterations and gap < config.msa_tolerance:
            converged = True
            break

    total_final = non_flow + ev_flow
    final_link_time = road_times(total_final, network, config)
    final_queue = queue_metrics(station_entries_full, network, config)
    result = AssignmentResult(
        config=config,
        network=network,
        od=od,
        total_flow=total_final,
        ev_flow=ev_flow,
        non_ev_flow=non_flow,
        link_time_min=final_link_time,
        station_metrics=final_queue,
        served_ev_by_class=served_ev_by_class,
        potential_ev_by_class=potential_ev_by_class,
        served_non_ev=served_non_ev,
        iterations=iteration,
        converged=converged,
        relative_gap=gap,
        ev_path_pool=fast_pool,
        ev_path_pool_summary=fast_pool_summary,
        ev_path_pool_refreshes=pool_refreshes,
    )
    result.hybrid_refinement_state = refinement_state
    result.hybrid_yen_summary = _hybrid_refinement_summary(refinement_state, od)
    result.hybrid_yen_records = _hybrid_refined_record_table(
        refinement_state, od, config
    )
    return result


def _write_hybrid_outputs(result: AssignmentResult, output_dir: Path) -> Dict[str, Path]:
    summary = getattr(result, "hybrid_yen_summary", pd.DataFrame())
    records = getattr(result, "hybrid_yen_records", pd.DataFrame())
    summary_path = output_dir / f"{result.config.scenario_id}_hybrid_yen_summary.csv"
    records_path = output_dir / f"{result.config.scenario_id}_hybrid_yen_refined_records.csv"
    settings_path = output_dir / f"{result.config.scenario_id}_hybrid_yen_settings.json"
    summary.to_csv(summary_path, index=False)
    records.to_csv(records_path, index=False)
    settings_path.write_text(
        json.dumps(HYBRID_YEN_SETTINGS.__dict__, indent=2, default=list),
        encoding="utf-8",
    )
    print("Hybrid tree-seeded Yen outputs:")
    print(f"  {summary_path}")
    print(f"  {records_path}")
    print(f"  {settings_path}")
    return {
        "hybrid_yen_summary": summary_path,
        "hybrid_yen_refined_records": records_path,
        "hybrid_yen_settings": settings_path,
    }


def run_scenario(config: ScenarioConfig, data_root: Path, output_dir: Path) -> AssignmentResult:
    network, od = load_inputs(data_root, config)
    result = perform_assignment(
        network,
        od,
        config,
        path_cache_root=output_dir / "hybrid_yen_cache",
    )
    _fast_base_write_outputs(result, output_dir)
    _write_hybrid_outputs(result, output_dir)
    return result


from dataclasses import dataclass as _opt_dataclass, replace as _dc_replace
from typing import FrozenSet as _FrozenSet, Any as _Any
import contextlib as _contextlib
import copy as _copy
import io as _io
import itertools as _itertools
import random as _random


DATA_ROOT = Path(r"/Users/wenchengbao/Chicago Network/flood_filtered_network_outputs")
OUTPUT_ROOT = DATA_ROOT / "ChicagoRegional_flood_kept_only_C3_C8_C9_budget_1p5M_outputs"

ANALYSIS_HOURS = 3.0
BATTERY_RANGE_MILES = 235.0
INITIAL_SOC_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
EV_CLASS_SHARES = (0.25, 0.25, 0.25, 0.25)
THETA_EV = 0.2
THETA_NON_EV = 0.2
ALPHA_BPR = 0.15
BETA_BPR = 4.0
DEMAND_SCALE = 1.0


BASELINE_OPERATIONAL_STATIONS = ("C3", "C8", "C9")
DAMAGED_STATIONS = ("C1", "C2", "C4", "C5", "C6", "C7")
PRIORITY_STATION_START_ORDER = ("C7", "C6", "C1", "C2", "C4", "C5")


MAX_INITIAL_STATIONS_PER_SEED = 2
PRIORITY_STATION_SEED_PACKAGES = (
    ("C3", "C7"),
    ("C6",),
    ("C1", "C7"),
    ("C2", "C6"),
)

BUDGET_USD = 1_500_000.0
ROAD_COST_PER_LANE_MILE_USD = 22_024.0
STATION_COST_PER_PORT_USD = 12_000.0
CAPACITY_PER_LANE_FOR_COST_VEHPH = 1900.0
MAX_LANES_FOR_COST = 5
EXCLUDE_CENTROID_CONNECTORS_FROM_REPAIR = True


QUEUE_WAIT_DISUTILITY_MULTIPLIER = 2.0
QUEUE_UTILIZATION_PENALTY_START = 0.85
QUEUE_UTILIZATION_PENALTY_SCALE_MIN = 5.0
QUEUE_UTILIZATION_PENALTY_POWER = 1.0
QUEUE_UTILIZATION_PENALTY_CAP_MIN = 1.0e6
QUEUE_UNSTABLE_LOADING_PENALTY_MIN = 1.0e6


EV_INITIAL_K_PATHS = 5
EV_PATH_REFRESH_EVERY = 0
EV_PREFER_DISTINCT_STATIONS = True
EV_PATH_MINHASH_SIZE = 8
EV_PATH_DIVERSITY_WEIGHT = 0.80
EV_PATH_COST_WEIGHT = 0.20
EV_PATH_RANDOM_SEED = 20260817
EV_DIRECT_CANDIDATE_TREES_SCREEN = 5
EV_DIRECT_CANDIDATE_TREES_FINAL = 10
EV_DIRECT_PERTURBATION_STRENGTH = 0.55


SCREEN_MSA_ITERATIONS = 12
SCREEN_MSA_MIN_ITERATIONS = 5
SCREEN_MSA_TOLERANCE = 2.0e-3
SCREEN_PLAN_LIMIT = 8
MAX_STATION_SEEDS = 8
TOP_ROAD_POOL_SIZE = 350
RANDOMIZED_GREEDY_CANDIDATES = 3
LOCAL_NEIGHBOR_LIMIT = 6
FINALIST_COUNT = 3

FINAL_MSA_ITERATIONS = 120
FINAL_MSA_MIN_ITERATIONS = 10
FINAL_MSA_TOLERANCE = 5.0e-4
FINAL_YEN_INITIAL_MAX_RECORDS = 300
FINAL_YEN_ENRICH_EVERY = 20
FINAL_YEN_ENRICH_TOP_M = 30

POLISH_WINNER = True
POLISH_MSA_ITERATIONS = 1200
POLISH_MSA_MIN_ITERATIONS = 20
POLISH_MSA_TOLERANCE = 1.0e-4
POLISH_YEN_INITIAL_MAX_RECORDS = 1000
POLISH_YEN_ENRICH_EVERY = 20
POLISH_YEN_ENRICH_TOP_M = 50


ROAD_CAPACITY_LOSS_PROXY_WEIGHT = 0.08
ROAD_CLOSED_LINK_PROXY_WEIGHT = 0.20
MIN_ROAD_IMPACT_SCORE = 1.0e-9


RANDOM_SEED = 20260817
PRINT_SCREEN_ASSIGNMENT_LOGS = False
PRINT_FINAL_ASSIGNMENT_LOGS = True
RUN_OPTIMIZATION = True


@_opt_dataclass(frozen=True)
class RepairPlan:
    repaired_stations: _FrozenSet[str] = frozenset()
    repaired_roads: _FrozenSet[str] = frozenset()

    def key(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (
            tuple(sorted(str(x).upper() for x in self.repaired_stations)),
            tuple(sorted(str(x) for x in self.repaired_roads)),
        )


@_opt_dataclass
class PlanEvaluation:
    plan: RepairPlan
    mode: str
    cost_usd: float
    served_total: float
    served_ev: float
    served_non_ev: float
    finite_system_time_min: float
    physical_in_vehicle_time_min: float
    physical_wait_vehicle_min: float
    physical_service_vehicle_min: float
    unstable_station_entries: float
    objective_key: Tuple[float, float, float, float]
    result: Optional[AssignmentResult] = None
    source: str = ""


@_opt_dataclass(frozen=True)
class SearchMode:
    name: str
    max_msa_iterations: int
    min_msa_iterations: int
    msa_tolerance: float
    tree_count: int
    hybrid_settings: HybridYenSettings
    print_logs: bool


def _screen_hybrid_settings() -> HybridYenSettings:
    return HybridYenSettings(
        enabled=False,
        initial_max_records=0,
        cumulative_demand_coverage=0.0,
        search_k=5,
        overlap_trigger=1.0,
        minimum_station_choices=1,
        corridor_expansions=(0,),
        allow_full_graph_fallback=False,
        max_stations_per_record=1,
        n_jobs=1,
        progress_every=1000,
        enrich_every=10**9,
        enrich_top_m=0,
        cache_initial_pool=False,
        rebuild_cache=False,
    )


def _final_hybrid_settings(polish: bool = False) -> HybridYenSettings:
    return HybridYenSettings(
        enabled=True,
        initial_max_records=(
            int(POLISH_YEN_INITIAL_MAX_RECORDS)
            if polish else int(FINAL_YEN_INITIAL_MAX_RECORDS)
        ),
        cumulative_demand_coverage=0.85,
        search_k=8,
        overlap_trigger=0.80,
        minimum_station_choices=2,
        corridor_expansions=(0, 1, 2),
        allow_full_graph_fallback=False,
        max_stations_per_record=3,
        n_jobs=8,
        progress_every=100,
        enrich_every=(
            int(POLISH_YEN_ENRICH_EVERY)
            if polish else int(FINAL_YEN_ENRICH_EVERY)
        ),
        enrich_top_m=(
            int(POLISH_YEN_ENRICH_TOP_M)
            if polish else int(FINAL_YEN_ENRICH_TOP_M)
        ),
        cache_initial_pool=True,
        rebuild_cache=False,
    )


SCREEN_MODE = SearchMode(
    name="screen",
    max_msa_iterations=int(SCREEN_MSA_ITERATIONS),
    min_msa_iterations=int(SCREEN_MSA_MIN_ITERATIONS),
    msa_tolerance=float(SCREEN_MSA_TOLERANCE),
    tree_count=int(EV_DIRECT_CANDIDATE_TREES_SCREEN),
    hybrid_settings=_screen_hybrid_settings(),
    print_logs=bool(PRINT_SCREEN_ASSIGNMENT_LOGS),
)
FINAL_MODE = SearchMode(
    name="final",
    max_msa_iterations=int(FINAL_MSA_ITERATIONS),
    min_msa_iterations=int(FINAL_MSA_MIN_ITERATIONS),
    msa_tolerance=float(FINAL_MSA_TOLERANCE),
    tree_count=int(EV_DIRECT_CANDIDATE_TREES_FINAL),
    hybrid_settings=_final_hybrid_settings(False),
    print_logs=bool(PRINT_FINAL_ASSIGNMENT_LOGS),
)
POLISH_MODE = SearchMode(
    name="polish",
    max_msa_iterations=int(POLISH_MSA_ITERATIONS),
    min_msa_iterations=int(POLISH_MSA_MIN_ITERATIONS),
    msa_tolerance=float(POLISH_MSA_TOLERANCE),
    tree_count=int(EV_DIRECT_CANDIDATE_TREES_FINAL),
    hybrid_settings=_final_hybrid_settings(True),
    print_logs=True,
)


def _base_assignment_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="chicago_flood_kept_C3_C8_C9_budget_1p5m",
        scenario_label="Flooding with budget-constrained repairs",
        use_flood_road_state=True,
        operational_stations=tuple(BASELINE_OPERATIONAL_STATIONS),
        analysis_hours=float(ANALYSIS_HOURS),
        battery_range_miles=float(BATTERY_RANGE_MILES),
        initial_soc_fractions=tuple(float(x) for x in INITIAL_SOC_FRACTIONS),
        ev_class_shares=tuple(float(x) for x in EV_CLASS_SHARES),
        theta_ev=float(THETA_EV),
        theta_non_ev=float(THETA_NON_EV),
        alpha_bpr=float(ALPHA_BPR),
        beta_bpr=float(BETA_BPR),
        max_msa_iterations=int(POLISH_MSA_ITERATIONS),
        min_msa_iterations=int(POLISH_MSA_MIN_ITERATIONS),
        msa_tolerance=float(POLISH_MSA_TOLERANCE),
        demand_scale=float(DEMAND_SCALE),
        print_all_links=False,
        progress_every=10,
        queue_wait_disutility_multiplier=float(QUEUE_WAIT_DISUTILITY_MULTIPLIER),
        queue_utilization_penalty_start=float(QUEUE_UTILIZATION_PENALTY_START),
        queue_utilization_penalty_scale_min=float(QUEUE_UTILIZATION_PENALTY_SCALE_MIN),
        queue_utilization_penalty_power=float(QUEUE_UTILIZATION_PENALTY_POWER),
        queue_utilization_penalty_cap_min=float(QUEUE_UTILIZATION_PENALTY_CAP_MIN),
        queue_unstable_loading_penalty_min=float(QUEUE_UNSTABLE_LOADING_PENALTY_MIN),
        ev_initial_k_paths=int(EV_INITIAL_K_PATHS),
        ev_path_refresh_every=int(EV_PATH_REFRESH_EVERY),
        ev_prefer_distinct_stations=bool(EV_PREFER_DISTINCT_STATIONS),
        ev_path_minhash_size=int(EV_PATH_MINHASH_SIZE),
        ev_path_diversity_weight=float(EV_PATH_DIVERSITY_WEIGHT),
        ev_path_cost_weight=float(EV_PATH_COST_WEIGHT),
        ev_path_random_seed=int(EV_PATH_RANDOM_SEED),
        ev_priority_station_id="",
        ev_require_priority_station_if_feasible=False,
        ev_direct_candidate_trees=int(EV_DIRECT_CANDIDATE_TREES_FINAL),
        ev_direct_perturbation_strength=float(EV_DIRECT_PERTURBATION_STRENGTH),
        ev_yen_search_k=8,
        ev_yen_n_jobs=8,
        ev_yen_rebuild_cache=False,
        ev_yen_progress_every=100,
    )


BASE_ASSIGNMENT_CONFIG = _base_assignment_config()


def _canonical_road_component_id(source_init: object, source_term: object) -> str:
    return f"{_normalize_node_label(source_init)}->{_normalize_node_label(source_term)}"


def load_fortification_master(
    data_root: Path,
) -> Tuple[NetworkData, ODData, pd.DataFrame, pd.DataFrame]:


    dry_loader_config = _dc_replace(
        BASE_ASSIGNMENT_CONFIG,
        scenario_id="fortification_master_loader",
        scenario_label="Fortification master loader",
        use_flood_road_state=False,
        operational_stations=tuple(f"C{i}" for i in range(1, 10)),
        max_msa_iterations=1,
        min_msa_iterations=1,
    )
    master, od = load_inputs(Path(data_root), dry_loader_config)

    raw_path = locate_local_csv(
        Path(data_root), "ChicagoRegional_net_flood_kept.csv"
    )
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    if len(raw) != len(master.links):
        raise RuntimeError(
            "Raw flood-kept network row count does not match the in-memory link table."
        )

    links = master.links.copy()
    raw_source_init = (
        raw["source_init_node"]
        if "source_init_node" in raw.columns else raw["init_node"]
    ).map(_normalize_node_label)
    raw_source_term = (
        raw["source_term_node"]
        if "source_term_node" in raw.columns else raw["term_node"]
    ).map(_normalize_node_label)
    source_link_type = pd.to_numeric(raw.get("link_type", 1), errors="coerce").fillna(1)
    source_flood_affected = _parse_bool_series(raw["flood_affected"])

    road_mask = links["type"].eq("road").to_numpy()
    component_ids = np.full(len(links), "", dtype=object)
    for index in np.where(road_mask)[0]:
        component_ids[index] = _canonical_road_component_id(
            raw_source_init.iloc[index], raw_source_term.iloc[index]
        )

    links["road_component_id"] = component_ids
    links["source_original_init_node"] = raw_source_init.to_numpy(dtype=str)
    links["source_original_term_node"] = raw_source_term.to_numpy(dtype=str)
    links["source_link_type"] = source_link_type.to_numpy(dtype=float)
    links["repair_candidate_flooded_road"] = (
        road_mask & source_flood_affected
    ).astype(np.int8)
    master.links = links

    road_rows = links[
        links["type"].eq("road")
        & links["repair_candidate_flooded_road"].eq(1)
    ].copy()
    if EXCLUDE_CENTROID_CONNECTORS_FROM_REPAIR:
        road_rows = road_rows[road_rows["source_link_type"] != 3]

    component_records: List[Dict[str, object]] = []
    for component_id, group in road_rows.groupby("road_component_id", sort=False):
        edge_indices = tuple(int(i) for i in group.index)
        length_miles = float(group["length_miles"].sum())
        lanes = int(group["inferred_lanes"].max())
        lane_miles = length_miles * lanes
        component_records.append({
            "road_component_id": str(component_id),
            "source_init_node": str(group["source_original_init_node"].iloc[0]),
            "source_term_node": str(group["source_original_term_node"].iloc[0]),
            "edge_count": len(edge_indices),
            "edge_indices": edge_indices,
            "length_miles": length_miles,
            "lanes": lanes,
            "lane_miles": lane_miles,
            "max_flood_depth_mm": float(group["source_flood_depth_mm"].max()),
            "minimum_capacity_retention_ratio": float(
                group["flood_capacity_retention_ratio"].min()
            ),
            "contains_closed_edge": bool(group["is_closed"].astype(bool).any()),
            "repair_cost_usd": float(
                ROAD_COST_PER_LANE_MILE_USD * lane_miles
            ),
        })
    road_components = pd.DataFrame(component_records)
    if road_components.empty:
        raise RuntimeError("No repairable flooded road components were identified.")
    road_components = road_components.sort_values(
        ["repair_cost_usd", "road_component_id"]
    ).reset_index(drop=True)

    stations = master.stations.copy()
    stations["station_id"] = stations["station_id"].astype(str).str.upper()
    stations["repair_cost_usd"] = (
        stations["k_ports"].astype(float) * STATION_COST_PER_PORT_USD
    )
    stations["initially_operational"] = stations["station_id"].isin(
        BASELINE_OPERATIONAL_STATIONS
    )
    stations["repair_candidate"] = stations["station_id"].isin(DAMAGED_STATIONS)
    stations["individually_budget_feasible"] = (
        stations["repair_cost_usd"] <= BUDGET_USD + 1.0e-9
    )
    return master, od, road_components, stations


def station_cost_map(station_table: pd.DataFrame) -> Dict[str, float]:
    return {
        str(row.station_id).upper(): float(row.repair_cost_usd)
        for row in station_table.itertuples(index=False)
        if bool(row.repair_candidate)
    }


def road_cost_map(road_table: pd.DataFrame) -> Dict[str, float]:
    return {
        str(row.road_component_id): float(row.repair_cost_usd)
        for row in road_table.itertuples(index=False)
    }


def repair_plan_cost(
    plan: RepairPlan,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
) -> float:
    return float(
        sum(float(road_costs[r]) for r in plan.repaired_roads)
        + sum(float(station_costs[s]) for s in plan.repaired_stations)
    )


def validate_repair_plan(
    plan: RepairPlan,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
) -> float:
    unknown_roads = sorted(set(plan.repaired_roads) - set(road_costs))
    unknown_stations = sorted(set(plan.repaired_stations) - set(station_costs))
    if unknown_roads:
        raise ValueError(f"Unknown repaired road components: {unknown_roads[:10]}")
    if unknown_stations:
        raise ValueError(f"Unknown repaired stations: {unknown_stations}")
    cost = repair_plan_cost(plan, road_costs, station_costs)
    if cost > BUDGET_USD + 1.0e-6:
        raise ValueError(
            f"Repair plan costs ${cost:,.2f}, exceeding budget ${BUDGET_USD:,.2f}."
        )
    return cost


def apply_repair_plan(
    master: NetworkData,
    plan: RepairPlan,
) -> NetworkData:

    network = _copy.copy(master)
    network.links = master.links.copy()
    network.stations = master.stations.copy()
    network.edge_t0_flood_min = master.edge_t0_flood_min.copy()
    network.edge_capacity_flood = master.edge_capacity_flood.copy()
    network.edge_is_closed = master.edge_is_closed.copy()
    network.operational_station_mask = master.operational_station_mask.copy()

    road = master.edge_type == "road"
    bypass = master.edge_type == "byp"
    entry = master.edge_type == "in"
    exit_edge_type = master.edge_type == "out"
    repaired_road_mask = (
        road
        & master.links["road_component_id"].astype(str).isin(
            plan.repaired_roads
        ).to_numpy()
    )


    network.edge_t0_flood_min[repaired_road_mask] = (
        master.edge_t0_dry_min[repaired_road_mask]
    )
    network.edge_capacity_flood[repaired_road_mask] = (
        master.edge_capacity_dry[repaired_road_mask]
    )
    network.edge_is_closed[repaired_road_mask] = 0

    base_open_road = (
        road
        & (master.edge_is_closed == 0)
        & np.isfinite(master.edge_t0_flood_min)
        & (master.edge_capacity_flood > 0.0)
    )
    active = np.zeros(len(master.links), dtype=np.bool_)
    active[road] = base_open_road[road] | repaired_road_mask[road]
    active[bypass] = True

    operational = {
        str(s).upper() for s in BASELINE_OPERATIONAL_STATIONS
    } | {str(s).upper() for s in plan.repaired_stations}
    station_mask = np.asarray(
        [str(sid).upper() in operational for sid in master.station_ids],
        dtype=np.bool_,
    )
    network.operational_station_mask = station_mask
    for station_index in range(len(master.station_ids)):
        entry_edge = int(master.station_entry_edge[station_index])
        exit_edge = int(master.station_exit_edge[station_index])
        bypass_edge = int(master.station_bypass_edge[station_index])
        active[entry_edge] = bool(station_mask[station_index])
        active[exit_edge] = bool(station_mask[station_index])
        active[bypass_edge] = True


    active[entry & ~active] = False
    active[exit_edge_type & ~active] = False
    network.active_edge = active
    network.travel_edge = active & (road | bypass)

    network.links["fortified_road"] = repaired_road_mask.astype(np.int8)
    network.links["fortified_station"] = (
        network.links["station_id"].astype(str).str.upper().isin(
            plan.repaired_stations
        ).astype(np.int8)
    )
    network.links["active_in_plan"] = active.astype(np.int8)
    network.stations["fortified"] = (
        network.stations["station_id"].astype(str).str.upper().isin(
            plan.repaired_stations
        ).astype(np.int8)
    )
    network.stations["operational_in_plan"] = (
        network.stations["station_id"].astype(str).str.upper().isin(
            operational
        ).astype(np.int8)
    )
    network.links["t0_flood_min"] = network.edge_t0_flood_min
    network.links["capacity_flood_vehph"] = network.edge_capacity_flood
    network.links["is_closed"] = network.edge_is_closed
    return network


def mode_config_for_plan(
    plan: RepairPlan,
    mode: SearchMode,
) -> ScenarioConfig:
    digest = hashlib.sha1(repr(plan.key()).encode("utf-8")).hexdigest()[:10]
    operational = tuple(sorted(
        set(BASELINE_OPERATIONAL_STATIONS) | set(plan.repaired_stations),
        key=lambda sid: int(str(sid)[1:]),
    ))
    return _dc_replace(
        BASE_ASSIGNMENT_CONFIG,
        scenario_id=f"flood_kept_C3_C8_C9_budget1500k_{mode.name}_{digest}",
        scenario_label=(
            f"Flooding with ${BUDGET_USD/1e6:.1f}M repairs ({mode.name})"
        ),
        use_flood_road_state=True,
        operational_stations=operational,
        max_msa_iterations=int(mode.max_msa_iterations),
        min_msa_iterations=int(mode.min_msa_iterations),
        msa_tolerance=float(mode.msa_tolerance),
        progress_every=(10 if mode.print_logs else mode.max_msa_iterations + 1),
        ev_direct_candidate_trees=int(mode.tree_count),
        ev_priority_station_id="",
        ev_require_priority_station_if_feasible=False,
    )


def _finite_evaluation_metrics(result: AssignmentResult) -> Dict[str, float]:
    road_mask = (result.network.edge_type == "road") & np.isfinite(result.link_time_min)
    road_vehicle_min = float(np.sum(
        result.total_flow[road_mask] * result.link_time_min[road_mask]
    ))
    entries = result.station_metrics.entries_period
    service_vehicle_min = float(np.sum(
        entries
        * result.network.station_service_min
        * result.network.operational_station_mask.astype(float)
    ))
    wait_vehicle_min = 0.0
    unstable_entries = 0.0
    for i in range(len(result.network.station_ids)):
        if not result.network.operational_station_mask[i]:
            continue
        q = float(entries[i])
        if q <= 0.0:
            continue
        wait_h = float(result.station_metrics.average_wait_hours[i])
        if math.isfinite(wait_h):
            wait_vehicle_min += q * wait_h * 60.0
        else:
            unstable_entries += q
            wait_vehicle_min += q * float(QUEUE_UNSTABLE_LOADING_PENALTY_MIN)
    return {
        "road_vehicle_min": road_vehicle_min,
        "service_vehicle_min": service_vehicle_min,
        "wait_vehicle_min": wait_vehicle_min,
        "unstable_station_entries": unstable_entries,
        "finite_system_time_min": (
            road_vehicle_min + service_vehicle_min + wait_vehicle_min
        ),
    }


def _lexicographic_key(
    served_total: float,
    served_ev: float,
    system_time: float,
    cost: float,
) -> Tuple[float, float, float, float]:


    return (
        round(float(served_total), 6),
        round(float(served_ev), 6),
        -float(system_time),
        -float(cost),
    )


_EVALUATION_CACHE: Dict[Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], str], PlanEvaluation] = {}


def evaluate_plan(
    *,
    master: NetworkData,
    od: ODData,
    plan: RepairPlan,
    mode: SearchMode,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
    cache_root: Path,
    source: str,
) -> PlanEvaluation:
    global HYBRID_YEN_SETTINGS
    cost = validate_repair_plan(plan, road_costs, station_costs)
    cache_key = (plan.key(), mode.name)
    if cache_key in _EVALUATION_CACHE:
        cached = _EVALUATION_CACHE[cache_key]
        if source and not cached.source:
            cached.source = source
        return cached

    plan_network = apply_repair_plan(master, plan)
    config = mode_config_for_plan(plan, mode)
    old_hybrid_settings = HYBRID_YEN_SETTINGS
    HYBRID_YEN_SETTINGS = mode.hybrid_settings
    plan_cache = cache_root / mode.name / hashlib.sha1(
        repr(plan.key()).encode("utf-8")
    ).hexdigest()[:16]
    plan_cache.mkdir(parents=True, exist_ok=True)

    try:
        if mode.print_logs:
            print(
                f"\nEvaluating {mode.name} plan: stations={sorted(plan.repaired_stations)}, "
                f"roads={len(plan.repaired_roads)}, cost=${cost:,.2f}"
            )
            result = perform_assignment(
                plan_network, od, config,
                path_cache_root=plan_cache / "hybrid_yen_cache",
            )
        else:
            buffer = _io.StringIO()
            with _contextlib.redirect_stdout(buffer):
                result = perform_assignment(
                    plan_network, od, config,
                    path_cache_root=plan_cache / "hybrid_yen_cache",
                )
    finally:
        HYBRID_YEN_SETTINGS = old_hybrid_settings

    finite = _finite_evaluation_metrics(result)
    served_ev = float(result.served_ev_by_class.sum())
    served_non = float(result.served_non_ev)
    served_total = served_ev + served_non
    key = _lexicographic_key(
        served_total, served_ev, finite["finite_system_time_min"], cost
    )
    evaluation = PlanEvaluation(
        plan=plan,
        mode=mode.name,
        cost_usd=cost,
        served_total=served_total,
        served_ev=served_ev,
        served_non_ev=served_non,
        finite_system_time_min=finite["finite_system_time_min"],
        physical_in_vehicle_time_min=finite["road_vehicle_min"],
        physical_wait_vehicle_min=finite["wait_vehicle_min"],
        physical_service_vehicle_min=finite["service_vehicle_min"],
        unstable_station_entries=finite["unstable_station_entries"],
        objective_key=key,
        result=result,
        source=source,
    )
    _EVALUATION_CACHE[cache_key] = evaluation
    return evaluation


def compute_road_impact_table(
    baseline_result: AssignmentResult,
    road_components: pd.DataFrame,
) -> pd.DataFrame:

    network = baseline_result.network
    flow = np.maximum(0.0, baseline_result.total_flow)
    H = float(baseline_result.config.analysis_hours)
    alpha = float(baseline_result.config.alpha_bpr)
    beta = float(baseline_result.config.beta_bpr)
    rows: List[Dict[str, object]] = []

    for component in road_components.itertuples(index=False):
        edges = np.asarray(component.edge_indices, dtype=np.int64)
        x = flow[edges]
        t0_dry = network.edge_t0_dry_min[edges]
        cap_dry = network.edge_capacity_dry[edges]
        t0_flood = network.edge_t0_flood_min[edges]
        cap_flood = network.edge_capacity_flood[edges]
        open_flood = (
            np.isfinite(t0_flood) & (cap_flood > 0.0)
            & (network.edge_is_closed[edges] == 0)
        )

        flood_time_same_flow = np.full(edges.size, np.inf, dtype=np.float64)
        if np.any(open_flood):
            rate = x[open_flood] / max(H, 1.0e-9)
            ratio = np.divide(
                rate, cap_flood[open_flood],
                out=np.zeros_like(rate), where=cap_flood[open_flood] > 0.0,
            )
            flood_time_same_flow[open_flood] = (
                t0_flood[open_flood]
                * (1.0 + alpha * np.power(ratio, beta))
            )

        rate_dry = x / max(H, 1.0e-9)
        dry_ratio = np.divide(
            rate_dry, cap_dry,
            out=np.zeros_like(rate_dry), where=cap_dry > 0.0,
        )
        dry_time_same_flow = t0_dry * (
            1.0 + alpha * np.power(dry_ratio, beta)
        )

        observed_saving = float(np.sum(
            x[open_flood]
            * np.maximum(
                0.0,
                flood_time_same_flow[open_flood]
                - dry_time_same_flow[open_flood],
            )
        ))
        cap_loss_ratio = np.divide(
            np.maximum(0.0, cap_dry - cap_flood),
            cap_dry,
            out=np.zeros_like(cap_dry),
            where=cap_dry > 0.0,
        )
        reference_flow = np.maximum(x, 0.03 * cap_dry * H)
        capacity_proxy = float(np.sum(
            reference_flow * t0_dry * cap_loss_ratio
        ))
        closed_proxy = 0.0
        if bool(component.contains_closed_edge):
            closed_proxy = float(np.sum(
                cap_dry * H * t0_dry
            ))
        impact = (
            observed_saving
            + float(ROAD_CAPACITY_LOSS_PROXY_WEIGHT) * capacity_proxy
            + float(ROAD_CLOSED_LINK_PROXY_WEIGHT) * closed_proxy
        )
        cost = float(component.repair_cost_usd)
        rows.append({
            **component._asdict(),
            "baseline_component_flow_veh_per_period": float(np.sum(x)),
            "same_flow_vehicle_min_saving": observed_saving,
            "capacity_loss_proxy_vehicle_min": capacity_proxy,
            "closed_link_proxy_vehicle_min": closed_proxy,
            "impact_score": impact,
            "impact_per_dollar": impact / max(cost, 1.0e-12),
        })

    table = pd.DataFrame(rows)
    table = table[table["impact_score"] > float(MIN_ROAD_IMPACT_SCORE)].copy()
    table = table.sort_values(
        ["impact_per_dollar", "impact_score", "road_component_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return table


def _station_proxy_value(row: _Any) -> float:


    return float(row.k_ports) / max(float(row.mean_service_min), 1.0e-9)


def generate_station_seed_plans(
    station_table: pd.DataFrame,
    station_costs: Mapping[str, float],
) -> List[Tuple[RepairPlan, str, float]]:

    candidates = [s for s in DAMAGED_STATIONS if s in station_costs]
    row_map = {
        str(row.station_id).upper(): row
        for row in station_table.itertuples(index=False)
    }
    baseline_set = {str(s).upper() for s in BASELINE_OPERATIONAL_STATIONS}
    print("\nPriority station warm-start audit:")
    print(f"  baseline operational stations={sorted(baseline_set)}")
    seeds: List[Tuple[RepairPlan, str, float]] = [
        (RepairPlan(), "no-station baseline seed", 0.0)
    ]
    seen_station_sets = {tuple()}

    def add_operational_guess(
        operational_guess: Sequence[str], source: str, priority_bonus: float
    ) -> None:
        requested = tuple(sorted({str(s).upper() for s in operational_guess}))
        repairs = tuple(sorted(set(requested) - baseline_set))
        if not repairs or repairs in seen_station_sets:
            return
        if len(repairs) > int(MAX_INITIAL_STATIONS_PER_SEED):
            return
        if any(s not in station_costs for s in repairs):
            return
        cost = sum(float(station_costs[s]) for s in repairs)
        if cost > BUDGET_USD + 1.0e-9:
            print(
                f"  operational guess {requested} -> repairs {repairs}: "
                f"${cost:,.0f} — skipped; exceeds ${BUDGET_USD:,.0f}"
            )
            return
        value = sum(_station_proxy_value(row_map[s]) for s in repairs)
        seeds.append((
            RepairPlan(repaired_stations=frozenset(repairs)),
            source + f" [repairs={'+'.join(repairs)}]",
            float(priority_bonus + value),
        ))
        seen_station_sets.add(repairs)
        print(
            f"  operational guess {requested} -> repairs {repairs}: "
            f"${cost:,.0f} — added; road budget remaining=${BUDGET_USD-cost:,.0f}"
        )

    for rank, package in enumerate(PRIORITY_STATION_SEED_PACKAGES):
        add_operational_guess(
            package,
            "requested operational station guess " + "+".join(package),
            priority_bonus=1.0e9 - 1.0e6 * rank,
        )

    for rank, sid in enumerate(PRIORITY_STATION_START_ORDER):
        add_operational_guess(
            (sid,),
            f"individual station repair seed {sid}",
            priority_bonus=1.0e8 - 1.0e5 * rank,
        )

    subset_candidates: List[Tuple[float, float, Tuple[str, ...]]] = []
    for size in range(1, min(int(MAX_INITIAL_STATIONS_PER_SEED), len(candidates)) + 1):
        for subset in _itertools.combinations(candidates, size):
            normalized = tuple(sorted(subset))
            if normalized in seen_station_sets:
                continue
            cost = sum(float(station_costs[s]) for s in normalized)
            if cost > BUDGET_USD + 1.0e-9:
                continue
            value = sum(_station_proxy_value(row_map[s]) for s in normalized)
            priority_bonus = 1.0e5 * len({"C7", "C6"}.intersection(normalized))
            subset_candidates.append((priority_bonus + value, -cost, normalized))
    subset_candidates.sort(reverse=True)
    for value, _, subset in subset_candidates:
        if len(seeds) >= int(MAX_STATION_SEEDS):
            break
        add_operational_guess(subset, "feasible parsimonious station subset", float(value))

    initial_sets = {frozenset(plan.repaired_stations) for plan, _, _ in seeds}


    required_feasible = {frozenset({"C7"}), frozenset({"C1", "C7"})}
    missing = required_feasible - initial_sets
    if missing:
        raise RuntimeError(f"Required feasible warm starts are missing: {missing}")
    if any("C3" in station_set for station_set in initial_sets):
        raise RuntimeError("Baseline station C3 was incorrectly charged as a repair.")
    if any(len(s) > int(MAX_INITIAL_STATIONS_PER_SEED) for s in initial_sets):
        raise RuntimeError("An initial station seed exceeds the configured size cap.")
    return seeds


def greedy_fill_roads(
    seed: RepairPlan,
    road_impact: pd.DataFrame,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
    strategy: str,
    rng: Optional[_random.Random] = None,
) -> RepairPlan:
    cost = repair_plan_cost(seed, road_costs, station_costs)
    selected = set(seed.repaired_roads)
    pool = road_impact.head(int(TOP_ROAD_POOL_SIZE)).copy()
    if strategy == "raw_impact":
        pool = pool.sort_values(
            ["impact_score", "impact_per_dollar"], ascending=False
        )
    elif strategy == "balanced":
        max_score = max(float(pool["impact_score"].max()), 1.0e-12)
        max_ratio = max(float(pool["impact_per_dollar"].max()), 1.0e-12)
        pool["_rank_value"] = (
            0.55 * pool["impact_score"] / max_score
            + 0.45 * pool["impact_per_dollar"] / max_ratio
        )
        pool = pool.sort_values("_rank_value", ascending=False)
    elif strategy == "randomized":
        if rng is None:
            rng = _random.Random(RANDOM_SEED)
        remaining = pool.sort_values("impact_per_dollar", ascending=False).to_dict("records")
        ordered: List[Dict[str, object]] = []
        while remaining:
            rcl = remaining[: min(8, len(remaining))]
            chosen = rng.choice(rcl)
            ordered.append(chosen)
            remaining.remove(chosen)
        pool = pd.DataFrame(ordered)
    else:
        pool = pool.sort_values(
            ["impact_per_dollar", "impact_score"], ascending=False
        )

    for row in pool.itertuples(index=False):
        rid = str(row.road_component_id)
        if rid in selected:
            continue
        rcost = float(road_costs[rid])
        if cost + rcost <= BUDGET_USD + 1.0e-9:
            selected.add(rid)
            cost += rcost
    return RepairPlan(
        repaired_stations=seed.repaired_stations,
        repaired_roads=frozenset(selected),
    )


def plan_surrogate_value(
    plan: RepairPlan,
    road_impact_map: Mapping[str, float],
    station_table: pd.DataFrame,
) -> float:
    station_rows = {
        str(row.station_id).upper(): row
        for row in station_table.itertuples(index=False)
    }
    road_value = sum(float(road_impact_map.get(r, 0.0)) for r in plan.repaired_roads)
    station_value = sum(
        1000.0 * _station_proxy_value(station_rows[s])
        for s in plan.repaired_stations
    )
    return road_value + station_value


def generate_initial_candidate_plans(
    station_seeds: Sequence[Tuple[RepairPlan, str, float]],
    road_impact: pd.DataFrame,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
    station_table: pd.DataFrame,
) -> List[Tuple[RepairPlan, str, float]]:
    rng = _random.Random(int(RANDOM_SEED))
    raw: List[Tuple[RepairPlan, str, float]] = []
    for seed, seed_source, _ in station_seeds:
        raw.append((seed, seed_source + " / station-only", 0.0))
        for strategy in ("ratio", "raw_impact", "balanced"):
            plan = greedy_fill_roads(
                seed, road_impact, road_costs, station_costs, strategy
            )
            raw.append((plan, f"{seed_source} / roads={strategy}", 0.0))
        for random_index in range(int(RANDOMIZED_GREEDY_CANDIDATES)):
            plan = greedy_fill_roads(
                seed, road_impact, road_costs, station_costs,
                "randomized", rng,
            )
            raw.append((
                plan,
                f"{seed_source} / randomized road fill {random_index+1}",
                0.0,
            ))

    impact_map = dict(zip(
        road_impact["road_component_id"].astype(str),
        road_impact["impact_score"].astype(float),
    ))
    dedup: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[RepairPlan, str, float]] = {}
    for plan, source, _ in raw:
        key = plan.key()
        value = plan_surrogate_value(plan, impact_map, station_table)
        current = dedup.get(key)
        if current is None or value > current[2]:
            dedup[key] = (plan, source, value)

    ordered = sorted(dedup.values(), key=lambda x: x[2], reverse=True)


    mandatory: List[Tuple[RepairPlan, str, float]] = []
    empty_item = next((
        item for item in ordered
        if not item[0].repaired_stations and not item[0].repaired_roads
    ), None)
    if empty_item is not None:
        mandatory.append(empty_item)

    best_by_station_set: Dict[Tuple[str, ...], Tuple[RepairPlan, str, float]] = {}
    for item in ordered:
        station_key = tuple(sorted(item[0].repaired_stations))
        if not station_key:
            continue
        if station_key not in best_by_station_set:
            best_by_station_set[station_key] = item

    station_set_order = [
        tuple(sorted(seed_plan.repaired_stations))
        for seed_plan, _, _ in station_seeds
        if seed_plan.repaired_stations
    ]
    for station_key in station_set_order:
        item = best_by_station_set.get(station_key)
        if item is not None:
            mandatory.append(item)

    selected: List[Tuple[RepairPlan, str, float]] = []
    seen = set()
    for item in mandatory + ordered:
        key = item[0].key()
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= int(SCREEN_PLAN_LIMIT):
            break
    return selected


def generate_local_neighbors(
    best: PlanEvaluation,
    road_impact: pd.DataFrame,
    road_costs: Mapping[str, float],
    station_costs: Mapping[str, float],
) -> List[Tuple[RepairPlan, str]]:
    plan = best.plan
    selected_roads = set(plan.repaired_roads)
    selected_stations = set(plan.repaired_stations)
    impact_map = dict(zip(
        road_impact["road_component_id"].astype(str),
        road_impact["impact_score"].astype(float),
    ))
    ratio_order = road_impact["road_component_id"].astype(str).tolist()
    low_selected = sorted(
        selected_roads,
        key=lambda rid: impact_map.get(rid, 0.0) / max(road_costs[rid], 1.0e-12),
    )[:4]
    high_unselected = [rid for rid in ratio_order if rid not in selected_roads][:16]
    candidates: List[Tuple[RepairPlan, str]] = []


    for old in low_selected:
        for new in high_unselected:
            roads = (selected_roads - {old}) | {new}
            neighbor = RepairPlan(frozenset(selected_stations), frozenset(roads))
            if repair_plan_cost(neighbor, road_costs, station_costs) <= BUDGET_USD + 1e-9:
                candidates.append((neighbor, f"swap road {old} -> {new}"))
                break
        if len(candidates) >= int(LOCAL_NEIGHBOR_LIMIT):
            break


    for station in PRIORITY_STATION_START_ORDER:
        if station not in station_costs or station in selected_stations:
            continue
        if station_costs[station] > BUDGET_USD + 1e-9:
            continue
        roads = set(selected_roads)
        stations = set(selected_stations) | {station}
        neighbor = RepairPlan(frozenset(stations), frozenset(roads))
        while (
            repair_plan_cost(neighbor, road_costs, station_costs) > BUDGET_USD + 1e-9
            and roads
        ):
            weakest = min(
                roads,
                key=lambda rid: impact_map.get(rid, 0.0) / max(road_costs[rid], 1e-12),
            )
            roads.remove(weakest)
            neighbor = RepairPlan(frozenset(stations), frozenset(roads))
        if repair_plan_cost(neighbor, road_costs, station_costs) <= BUDGET_USD + 1e-9:
            candidates.append((neighbor, f"add station {station} with budget repair"))
        if len(candidates) >= int(LOCAL_NEIGHBOR_LIMIT):
            break

    dedup: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[RepairPlan, str]] = {}
    for neighbor, source in candidates:
        if neighbor.key() != plan.key():
            dedup[neighbor.key()] = (neighbor, source)
    return list(dedup.values())[: int(LOCAL_NEIGHBOR_LIMIT)]


def evaluation_table(evaluations: Sequence[PlanEvaluation]) -> pd.DataFrame:
    rows = []
    for rank, evaluation in enumerate(
        sorted(evaluations, key=lambda e: e.objective_key, reverse=True), start=1
    ):
        rows.append({
            "rank_within_available_evaluations": rank,
            "mode": evaluation.mode,
            "source": evaluation.source,
            "repair_cost_usd": evaluation.cost_usd,
            "budget_remaining_usd": BUDGET_USD - evaluation.cost_usd,
            "repaired_station_count": len(evaluation.plan.repaired_stations),
            "repaired_stations": ";".join(sorted(evaluation.plan.repaired_stations)),
            "repaired_road_component_count": len(evaluation.plan.repaired_roads),
            "served_total": evaluation.served_total,
            "served_ev": evaluation.served_ev,
            "served_non_ev": evaluation.served_non_ev,
            "finite_system_time_min": evaluation.finite_system_time_min,
            "physical_in_vehicle_time_min": evaluation.physical_in_vehicle_time_min,
            "physical_wait_vehicle_min": evaluation.physical_wait_vehicle_min,
            "physical_service_vehicle_min": evaluation.physical_service_vehicle_min,
            "unstable_station_entries": evaluation.unstable_station_entries,
        })
    return pd.DataFrame(rows)


def save_best_plan_outputs(
    best: PlanEvaluation,
    road_impact: pd.DataFrame,
    road_components: pd.DataFrame,
    station_table: pd.DataFrame,
    all_evaluations: Sequence[PlanEvaluation],
    output_dir: Path,
) -> Dict[str, Path]:
    if best.result is None:
        raise RuntimeError("Best plan has no assignment result.")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = best.result


    result.config = _dc_replace(
        result.config,
        scenario_id="chicago_flood_kept_C3_C8_C9_budget_1p5m_best_plan",
        scenario_label="Flood-kept network with C3/C8/C9 baseline and best repairs under $1.5M",
    )
    base_outputs = _fast_base_write_outputs(result, output_dir)
    hybrid_outputs = _write_hybrid_outputs(result, output_dir)

    selected_roads = road_components[
        road_components["road_component_id"].astype(str).isin(
            best.plan.repaired_roads
        )
    ].copy()
    selected_roads = selected_roads.merge(
        road_impact[[
            "road_component_id", "baseline_component_flow_veh_per_period",
            "same_flow_vehicle_min_saving", "capacity_loss_proxy_vehicle_min",
            "closed_link_proxy_vehicle_min", "impact_score", "impact_per_dollar",
        ]],
        on="road_component_id", how="left",
    ).sort_values("impact_score", ascending=False)
    selected_stations = station_table[
        station_table["station_id"].isin(best.plan.repaired_stations)
    ].copy()
    selected_stations["cost_per_port_usd"] = STATION_COST_PER_PORT_USD

    road_path = output_dir / "selected_road_repairs.csv"
    station_path = output_dir / "selected_station_repairs.csv"
    eval_path = output_dir / "candidate_plan_evaluations.csv"
    impact_path = output_dir / "road_impact_ranking.csv"
    summary_path = output_dir / "optimization_summary.json"
    selected_roads.to_csv(road_path, index=False)
    selected_stations.to_csv(station_path, index=False)
    evaluation_table(all_evaluations).to_csv(eval_path, index=False)
    road_impact.to_csv(impact_path, index=False)

    payload = {
        "budget_usd": BUDGET_USD,
        "road_cost_per_lane_mile_usd": ROAD_COST_PER_LANE_MILE_USD,
        "station_cost_per_port_usd": STATION_COST_PER_PORT_USD,
        "initial_soc_fractions": list(INITIAL_SOC_FRACTIONS),
        "ev_class_shares": list(EV_CLASS_SHARES),
        "baseline_operational_stations": list(BASELINE_OPERATIONAL_STATIONS),
        "repaired_stations": sorted(best.plan.repaired_stations),
        "repaired_road_components": sorted(best.plan.repaired_roads),
        "repair_cost_usd": best.cost_usd,
        "budget_remaining_usd": BUDGET_USD - best.cost_usd,
        "served_total": best.served_total,
        "served_ev": best.served_ev,
        "served_non_ev": best.served_non_ev,
        "finite_system_time_min": best.finite_system_time_min,
        "physical_in_vehicle_time_min": best.physical_in_vehicle_time_min,
        "physical_wait_vehicle_min": best.physical_wait_vehicle_min,
        "physical_service_vehicle_min": best.physical_service_vehicle_min,
        "unstable_station_entries": best.unstable_station_entries,
        "priority_operational_station_guesses": [
            list(x) for x in PRIORITY_STATION_SEED_PACKAGES
        ],
        "selection_note": (
            "C3 is already operational, so C3+C7 is charged as the C7 repair only. "
            "C6 and C2+C6 remain requested guesses but are skipped under the $1.5M "
            "budget because full-station repairs are binary; partial-port repair is not used."
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        **base_outputs,
        **hybrid_outputs,
        "selected_roads": road_path,
        "selected_stations": station_path,
        "candidate_evaluations": eval_path,
        "road_impact_ranking": impact_path,
        "optimization_summary": summary_path,
    }


def run_budget_repair_search() -> Dict[str, _Any]:
    output_dir = OUTPUT_ROOT / "chicago_flood_kept_C3_C8_C9_budget_1p5m_repair_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Notebook working directory: {Path.cwd()}")
    print(f"Data search root: {DATA_ROOT.resolve()}")
    print(f"Optimization output directory: {output_dir.resolve()}")
    print(f"Budget: ${BUDGET_USD:,.0f}")
    print(
        f"Costs: ${ROAD_COST_PER_LANE_MILE_USD:,.0f}/lane-mile; "
        f"${STATION_COST_PER_PORT_USD:,.0f}/port"
    )
    print(
        f"EV SoC classes: {INITIAL_SOC_FRACTIONS}; shares={EV_CLASS_SHARES}; "
        f"full range={BATTERY_RANGE_MILES:.1f} miles"
    )

    master, od, road_components, station_table = load_fortification_master(DATA_ROOT)
    road_costs = road_cost_map(road_components)
    station_costs = station_cost_map(station_table)

    station_audit = station_table[[
        "station_id", "k_ports", "mean_service_min", "repair_cost_usd",
        "initially_operational", "repair_candidate", "individually_budget_feasible",
    ]].copy()
    station_audit_path = output_dir / "station_repair_cost_audit.csv"
    station_audit.to_csv(station_audit_path, index=False)
    print("\nStation repair costs:")
    print(station_audit.to_string(index=False))

    cache_root = output_dir / "plan_evaluation_cache"
    baseline_plan = RepairPlan()
    baseline_eval = evaluate_plan(
        master=master, od=od, plan=baseline_plan, mode=SCREEN_MODE,
        road_costs=road_costs, station_costs=station_costs,
        cache_root=cache_root, source="flood baseline C3-C8-C9",
    )
    print(
        f"\nBaseline screen: served EV={baseline_eval.served_ev:,.3f}; "
        f"system time={baseline_eval.finite_system_time_min:,.3f} min"
    )

    road_impact = compute_road_impact_table(
        baseline_eval.result, road_components
    )
    print("\nTop 20 road components by impact per dollar:")
    print(road_impact.head(20)[[
        "road_component_id", "length_miles", "lanes", "repair_cost_usd",
        "baseline_component_flow_veh_per_period", "impact_score",
        "impact_per_dollar",
    ]].to_string(index=False))

    station_seeds = generate_station_seed_plans(station_table, station_costs)
    candidate_plans = generate_initial_candidate_plans(
        station_seeds, road_impact, road_costs, station_costs, station_table
    )
    print(f"\nScreening {len(candidate_plans)} unique initial plans...")
    screened: List[PlanEvaluation] = [baseline_eval]
    for index, (plan, source, surrogate) in enumerate(candidate_plans, start=1):
        if plan.key() == baseline_plan.key():
            continue
        cost = repair_plan_cost(plan, road_costs, station_costs)
        print(
            f"  screen {index}/{len(candidate_plans)}: {source}; "
            f"stations={sorted(plan.repaired_stations)}; roads={len(plan.repaired_roads)}; "
            f"cost=${cost:,.0f}"
        )
        screened.append(evaluate_plan(
            master=master, od=od, plan=plan, mode=SCREEN_MODE,
            road_costs=road_costs, station_costs=station_costs,
            cache_root=cache_root, source=source,
        ))

    best_screen = max(screened, key=lambda e: e.objective_key)
    print(
        f"\nBest initial screen plan: stations={sorted(best_screen.plan.repaired_stations)}, "
        f"roads={len(best_screen.plan.repaired_roads)}, cost=${best_screen.cost_usd:,.0f}, "
        f"served EV={best_screen.served_ev:,.3f}"
    )

    neighbors = generate_local_neighbors(
        best_screen, road_impact, road_costs, station_costs
    )
    print(f"Evaluating {len(neighbors)} local add/swap neighbors...")
    for neighbor, source in neighbors:
        screened.append(evaluate_plan(
            master=master, od=od, plan=neighbor, mode=SCREEN_MODE,
            road_costs=road_costs, station_costs=station_costs,
            cache_root=cache_root, source=source,
        ))

    screen_ranked = sorted(
        {e.plan.key(): e for e in screened}.values(),
        key=lambda e: e.objective_key,
        reverse=True,
    )
    finalists = screen_ranked[: max(1, int(FINALIST_COUNT))]
    print(f"\nRunning selective-Yen final evaluation for {len(finalists)} finalists...")
    final_evaluations: List[PlanEvaluation] = []
    for index, screen_eval in enumerate(finalists, start=1):
        print(
            f"\nFinalist {index}/{len(finalists)} from {screen_eval.source}: "
            f"stations={sorted(screen_eval.plan.repaired_stations)}, "
            f"roads={len(screen_eval.plan.repaired_roads)}, "
            f"cost=${screen_eval.cost_usd:,.0f}"
        )
        final_evaluations.append(evaluate_plan(
            master=master, od=od, plan=screen_eval.plan, mode=FINAL_MODE,
            road_costs=road_costs, station_costs=station_costs,
            cache_root=cache_root, source=screen_eval.source,
        ))
    best_final = max(final_evaluations, key=lambda e: e.objective_key)

    if POLISH_WINNER:
        print("\nPolishing the winning plan with the full MSA/Yen settings...")
        best = evaluate_plan(
            master=master, od=od, plan=best_final.plan, mode=POLISH_MODE,
            road_costs=road_costs, station_costs=station_costs,
            cache_root=cache_root, source=best_final.source + " / polished",
        )
    else:
        best = best_final

    all_evaluations = list(_EVALUATION_CACHE.values())
    outputs = save_best_plan_outputs(
        best, road_impact, road_components, station_table,
        all_evaluations, output_dir,
    )
    print("\n" + "=" * 96)
    print("BEST BUDGET-FEASIBLE REPAIR PLAN")
    print("=" * 96)
    print(f"Repaired stations: {sorted(best.plan.repaired_stations)}")
    print(f"Repaired road components: {len(best.plan.repaired_roads)}")
    print(f"Repair cost: ${best.cost_usd:,.2f}")
    print(f"Budget remaining: ${BUDGET_USD-best.cost_usd:,.2f}")
    print(f"Served EV: {best.served_ev:,.6f}")
    print(f"Served non-EV: {best.served_non_ev:,.6f}")
    print(f"Finite system-time objective: {best.finite_system_time_min:,.6f} min")
    print("Created outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return {
        "best_plan": best.plan,
        "best_evaluation": best,
        "best_result": best.result,
        "road_impact": road_impact,
        "road_components": road_components,
        "station_costs": station_table,
        "plan_evaluations": evaluation_table(all_evaluations),
        "outputs": outputs,
        "output_dir": output_dir,
    }


_DEFAULT_OUTPUT_FOLDER = OUTPUT_ROOT.name


def run(data_root=None, output_root=None):
    global DATA_ROOT, OUTPUT_ROOT
    DATA_ROOT = Path.cwd().resolve() if data_root is None else Path(data_root).expanduser().resolve()
    OUTPUT_ROOT = DATA_ROOT / _DEFAULT_OUTPUT_FOLDER if output_root is None else Path(output_root).expanduser().resolve()
    if NUMBA_AVAILABLE:
        try:
            import numba as _numba_module
            print(f"Numba acceleration: ON (Numba {_numba_module.__version__}, NumPy {np.__version__})")
        except Exception:
            print("Numba acceleration: ON")
    else:
        print(f"WARNING: Numba acceleration unavailable: {NUMBA_IMPORT_ERROR}")
    optimization_output = run_budget_repair_search()
    best_plan = optimization_output["best_plan"]
    best_evaluation = optimization_output["best_evaluation"]
    best_result = optimization_output["best_result"]
    road_impact_results = optimization_output["road_impact"]
    road_components = optimization_output["road_components"]
    station_repair_costs = optimization_output["station_costs"]
    plan_evaluations = optimization_output["plan_evaluations"]
    link_results = build_link_table(best_result)
    station_results = build_station_table(best_result)
    summary_results = pd.DataFrame([compute_time_summary(best_result)])
    selected_road_repairs = road_components[
        road_components["road_component_id"].isin(best_plan.repaired_roads)
    ].copy()
    selected_station_repairs = station_repair_costs[
        station_repair_costs["station_id"].isin(best_plan.repaired_stations)
    ].copy()
    try:
        from IPython.display import display
        print("\nOptimization plan evaluations:")
        display(plan_evaluations)
        print("\nBest-plan summary:")
        display(summary_results)
        print("\nBest-plan station metrics:")
        display(station_results)
        print("\nSelected station repairs:")
        display(selected_station_repairs)
        print("\nTop selected road repairs:")
        display(selected_road_repairs.head(50))
        print("\nFirst 20 best-plan link flows:")
        display(link_results.head(20))
    except Exception:
        print(plan_evaluations.to_string(index=False))
        print(summary_results.to_string(index=False))
    return {
        **optimization_output,
        "best_evaluation": best_evaluation,
        "road_impact_results": road_impact_results,
        "link_results": link_results,
        "station_results": station_results,
        "summary_results": summary_results,
        "selected_road_repairs": selected_road_repairs,
        "selected_station_repairs": selected_station_repairs,
    }
