# Perception Services

Minimal perception service stack for robotics, focused on:

- **OWLv2** object detection
- **SAM** segmentation
- **Contact-GraspNet** grasp proposal API

The project uses **[`pixi`](https://pixi.sh/)** for reproducible environments and task management.

## Default Ports

- `4000` → OWLv2
- `4001` → SAM
- `4003` → Contact-GraspNet

## Quick Start

### 1) Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2) Bootstrap environments + third-party dependency

```bash
bash service/setup_all.sh
```

This sets up pixi envs, clones:

- `https://github.com/NVlabs/contact_graspnet.git` into `third_party/contact_graspnet`

and auto-downloads the Contact-GraspNet checkpoint if missing.

### 3) Optional: compile Contact-GraspNet PointNet ops

```bash
pixi run -e grasp compile-grasp-ops
```

## Run Services

### Run full stack locally

```bash
pixi run stack
```

### Run services individually

```bash
pixi run owl-server
pixi run sam-server
pixi run -e grasp grasp-server
```

Combined stack logs are written to `service/logs/`.

## Slurm Deployment (crane7)

Use the included launcher:

```bash
sbatch service/run_det_seg_grasp_crane7.sh
```

Current Slurm settings:

```text
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane7
```

## Health Checks

```bash
curl http://<host>:4000/healthz
curl http://<host>:4001/healthz
curl http://<host>:4003/healthz
```

## Smoke Tests

Tests are in `tests/`:

```bash
pixi run python tests/test_owlv2.py --server-url http://<host>:4000
pixi run python tests/test_sam.py --server-url http://<host>:4001
pixi run python tests/test_grasp.py --server-url http://<host>:4003
```

## Project Structure

```text
.
├── apis/                              # client-side API wrappers
├── service/
│   ├── owl_vit/                       # OWLv2 server
│   ├── sam/                           # SAM server
│   ├── grasp/                         # Contact-GraspNet wrapper + checkpoint bootstrap
│   ├── start_det_seg_grasp.sh         # multi-service local launcher
│   └── run_det_seg_grasp_crane7.sh    # Slurm launcher (crane7)
├── tests/                             # smoke test scripts
├── third_party/
│   └── contact_graspnet/              # upstream clone (kept unmodified)
└── pixi.toml                          # environments + tasks
```

## Notes

- Upstream `third_party/contact_graspnet` is used as cloned code and is not modified.
- Integration logic lives in this repo (`service/` launchers/wrappers and `apis/` clients).
