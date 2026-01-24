# LTV

Quick guide for environment setup and running the main experiment script.

## Environment Setup

1) Run the setup script:
```bash
./setup.sh
```

2) Activate the virtual environment (if not already active):
```bash
source .venv/bin/activate
```

Notes:
- `setup.sh` creates `.venv`, upgrades pip, and installs `requirements.txt`.
- The default install uses PyPI (CPU wheels). If you need CUDA wheels, install the correct torch build manually after setup.

## Run the Experiment

The main runner is `run/run_our_ltv.sh`. It calls `run/run_our_ltv.py` with the config file.

```bash
./run/run_our_ltv.sh
```

If you want to run the Python script directly:
```bash
python run/run_our_ltv.py --config_path config/config_our_litv.py
```

## Config

Edit experiment settings here:
- `config/config_our_litv.py`

Key flags to check:
- `run_baseline` (zero-shot/few-shot)
- `run_ltv` (LTV adaptive task vector)
*** End Patch
