# Chicago Flood-Resilient EV Transportation Network

This repository contains the data, Python implementation, and precomputed results for a ChicagoRegional case study of joint road and electric-vehicle charging-infrastructure fortification under flooding.

The model represents a three-hour EV/non-EV traffic-assignment period and combines flood-adjusted road conditions, road congestion, EV driving-range feasibility, charging-station queues, stochastic route and station choice, and budget-constrained infrastructure fortification.

## Repository structure

```text
.
├── non_flood.py
├── flood_only.py
├── 1.5.py
├── 5.py
├── 15.py
├── 25.py
├── 45.py
│
├── chicago_model/
│   ├── __init__.py
│   ├── cli.py
│   ├── no_flood_core.py
│   ├── flood_only_core.py
│   ├── budget_1p5_core.py
│   ├── budget_5_core.py
│   ├── budget_15_core.py
│   ├── budget_25_core.py
│   └── budget_45_core.py
│
├── no-flood-result/
├── flood_only_result/
├── 1.5M/
├── 5M/
├── 15M/
├── 25M/
├── 45M/
│
├── ChicagoRegional_net_flood_kept.csv
├── ChicagoRegional_node_flood_kept.csv
├── ChicagoRegional_EV_flood_kept_matrix.csv
├── ChicagoRegional_nonEV_flood_kept_matrix.csv
├── ChicagoRegional_EV_flood_kept.csv
└── clustering_sites_with_group.csv
```

### Source code

The short Python files in the repository root are scenario entry points. The reusable model implementation, numerical routines, queueing functions, path-generation procedures, and optimization methods are stored in `chicago_model/`.

The files `1.5.py`, `5.py`, `15.py`, `25.py`, and `45.py` are intended to be executed directly. Because their filenames begin with numbers, they should not be imported as ordinary Python modules.

### Precomputed results

The following directories contain outputs from model runs that have already been completed:

| Result directory | Corresponding entry script | Scenario |
|---|---|---|
| `no-flood-result/` | `non_flood.py` | No-flood baseline on the flood-kept topology |
| `flood_only_result/` | `flood_only.py` | Flooded network with no fortification |
| `1.5M/` | `1.5.py` | Flood fortification with a $1.5 million budget |
| `5M/` | `5.py` | Flood fortification with a $5 million budget |
| `15M/` | `15.py` | Flood fortification with a $15 million budget |
| `25M/` | `25.py` | Flood fortification with a $25 million budget |
| `45M/` | `45.py` | Flood fortification with a $45 million budget |

These folders are archived computational results, not source-code folders, and they are not required for executing the model. Depending on the scenario, they may contain link flows, charging-station metrics, EV completion summaries, selected road and station repairs, optimization histories, checkpoints, and path-set diagnostics.

## Model overview

The fixed no-flood and flood-only scenarios solve the lower-level network assignment problem. The budget scenarios additionally solve an upper-level road and charging-station fortification problem.

The implementation includes:

- shared EV and non-EV road congestion using the Bureau of Public Roads travel-time function;
- flood-adjusted road speeds and capacities;
- EV initial state-of-charge classes and a 235-mile full-charge driving range;
- energy-feasible direct and charging paths;
- multi-server `M/G/K` charging-station queues;
- hybrid tree-seeded selective Yen path generation with up to five EV alternatives;
- logit route and charging-station choice;
- the method of successive averages for the lower-level assignment; and
- a population-based Benders-inspired hyper-matheuristic for budget-constrained fortification.

For charging-required EV trips, the path-generation procedure prioritizes feasible alternatives through distinct charging stations before adding repeated alternatives through a station that is already represented. EV trips that can reach their destination directly are not routed through a charging station.

## Main experiment settings

| Setting | Value |
|---|---:|
| Analysis window | 3 hours |
| EV full-charge range | 235 miles |
| Initial EV state of charge | 10%, 20%, 30%, and 40% |
| Share of each state-of-charge class | 25% |
| Flood-scenario stations initially operational | C3, C8, and C9 |
| Road fortification cost | $22,024 per lane-mile |
| Charging-station fortification cost | $12,000 per port |

Road lengths are measured in miles and road speeds in miles per hour. The code recomputes dry free-flow road time as

```text
60 × length_miles / speed_mph
```

and infers between one and five lanes from source capacity using 1,900 vehicles per hour per lane.

## Required input files

The scenario scripts read the following primary inputs from the directory specified by `--data-root`:

```text
ChicagoRegional_net_flood_kept.csv
ChicagoRegional_node_flood_kept.csv
ChicagoRegional_EV_flood_kept_matrix.csv
ChicagoRegional_nonEV_flood_kept_matrix.csv
```

The node file also supplies charging-station port counts and charging-time information. The additional CSV files in the repository provide supporting or alternative representations of the EV demand and charging-site clusters.

## Installation

Python 3.10 or later is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required numerical packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy
```

Optional acceleration and optimization packages are:

```bash
python -m pip install numba
python -m pip install gurobipy
```

Numba is optional; the code can fall back to the corresponding Python implementation when Numba is unavailable. Gurobi is used when available for selected upper-level intensification steps, with non-Gurobi fallback logic retained in the budget modules.

## Running the scenarios

Run the commands from the repository root so that the default data directory is the folder containing the input CSV files.

### No-flood baseline

```bash
python non_flood.py
```

### Flooding with no fortification

```bash
python flood_only.py
```

### Budget-constrained fortification

```bash
python 1.5.py
python 5.py
python 15.py
python 25.py
python 45.py
```

## Command-line options

Every entry script accepts the same two optional arguments:

```text
--data-root PATH
--output-root PATH
```

For example:

```bash
python 25.py \
  --data-root . \
  --output-root ./rerun_outputs/25M
```

- `--data-root` identifies the directory containing the network and OD files.
- `--output-root` identifies the parent directory for newly generated results.

When both options are omitted, the current working directory is used as the data root and each module uses its scenario-specific default output directory.

To preserve the archived results already stored in `no-flood-result/`, `flood_only_result/`, and the budget folders, direct new runs to a separate location such as `rerun_outputs/`.

## Interpreting the outputs

Common outputs include:

- total, EV, and non-EV flow on each link;
- road travel times and vehicle-minutes;
- charging-station arrivals, utilization, waiting time, and total waiting time;
- EV trip completion by initial state of charge;
- scenario-level CSV and JSON summaries;
- selected fortified roads and charging stations;
- BIHMH generation histories and current-best checkpoints; and
- hybrid Yen path-pool summaries and refinement records.

The reported average in-vehicle time is calculated over travelers who complete their trips. Charging service time and charging-station waiting time are reported separately from road in-vehicle time.

## Reproducibility notes

- Run one scenario at a time, particularly for the larger budget cases, because the optimization and path-generation routines can require substantial memory.
- Restarting the Python or Jupyter kernel before a large budget run prevents arrays from earlier runs from remaining in memory.
- Budget-case checkpoints preserve the latest completed incumbent solution if execution is interrupted.
- Small numerical differences may occur across Python, SciPy, Numba, and parallel-execution environments.

## Associated study

This repository accompanies the study:

> **A Bilevel Optimization Framework for Flood Resilient Electric Vehicle Infrastructure on Urban Transport Networks**  
> Wencheng Bao and Eleftheria Kontou

Please cite the associated manuscript when using the model, data preparation, or computational results from this repository.
