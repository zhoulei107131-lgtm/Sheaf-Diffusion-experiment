# Sheaf-Diffusion-experiment

This project packages a small SPD transport experiment into a runnable Python module.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training

```bash
python -m spd_transport.train --steps 50
```

## Run tests

```bash
pytest -q
```
