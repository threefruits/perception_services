# Perception Services

Lightweight service host for robotics perception with three focused components:

- **OWLv2** object detection
- **SAM** segmentation
- **Contact-GraspNet** grasp proposal service

This repo is designed around **[`pixi`](https://pixi.sh/)** for reproducible environments and simple service startup.

## Services

Default ports:

- `4000` → OWLv2
- `4001` → SAM
- `4003` → Contact-GraspNet

All three can be started together or independently.

## Quick Start

### 1) Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2) Bootstrap the workspace

```bash
bash service/setup_all.sh
```

This does the following:

- installs project environments
- clones `third_party/contact_graspnet` if needed
- auto-downloads Contact-GraspNet checkpoints if missing

### 3) Compile Contact-GraspNet PointNet ops

```bash
pixi run -e grasp compile-grasp-ops
```

## Run Services

### Run all services

```bash
pixi run stack
```

### Run individually

```bash
pixi run owl-server
pixi run sam-server
pixi run -e grasp grasp-server
```

Logs from combined startup are written to `service/logs/`.

## Checkpoint Management

Checkpoint bootstrap for Contact-GraspNet is automatic during grasp startup.

Manual trigger:

```bash
pixi run -e grasp ensure-grasp-checkpoint
```

Useful overrides:

- `CONTACT_GRASPNET_CHECKPOINT_DIR` to point to a custom checkpoint path
- `CONTACT_GRASPNET_CHECKPOINT_URL` to use a custom download source

## Port Overrides

You can override ports when starting the full stack:

```bash
OWLV2_PORT=4100 SAM_PORT=4101 GRASP_PORT=4103 pixi run stack
```

## Slurm Usage

Default Slurm launcher:

```bash
sbatch service/run_det_seg_grasp_pi_crane6.sh
```

Default resource request in that script:

```text
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane6
```

To run on a different node and custom ports:

```bash
OWLV2_PORT=4000 SAM_PORT=4001 GRASP_PORT=4002 \
sbatch -w crane7 --export=ALL,OWLV2_PORT=4000,SAM_PORT=4001,GRASP_PORT=4002 \
service/run_det_seg_grasp_pi_crane6.sh
```

## API Health Checks

```bash
curl http://<host>:4000/healthz
curl http://<host>:4001/healthz
curl http://<host>:4003/healthz
```

## Project Structure

```text
.
├── apis/                         # client-side API wrappers
├── service/
│   ├── owl_vit/                  # OWLv2 server
│   ├── sam/                      # SAM server
│   ├── grasp/                    # grasp server wrappers/checkpoint bootstrap
│   ├── start_det_seg_grasp.sh    # local multi-service launcher
│   └── run_det_seg_grasp_pi_crane6.sh  # Slurm launcher
├── third_party/
│   └── contact_graspnet/         # upstream clone (kept unmodified)
└── pixi.toml                     # environments + tasks
```

## Troubleshooting

- If `grasp` fails on startup, check:
  - checkpoint exists under `third_party/contact_graspnet/checkpoints/...`
  - PointNet ops were compiled: `pixi run -e grasp compile-grasp-ops`
  - `service/logs/grasp.log`
- If OWLv2/SAM fail, inspect:
  - `service/logs/owlv2.log`
  - `service/logs/sam.log`

## Notes

- The `third_party/contact_graspnet` clone is treated as upstream code.
- Integration behavior is implemented in this repo (launcher scripts, pixi tasks, wrappers), not by modifying upstream sources.
