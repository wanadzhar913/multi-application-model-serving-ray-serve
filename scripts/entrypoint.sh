#!/bin/bash
uv run -- ray start --head --dashboard-host=0.0.0.0 --include-dashboard=True &
sleep 10
uv run -- serve run config.yaml