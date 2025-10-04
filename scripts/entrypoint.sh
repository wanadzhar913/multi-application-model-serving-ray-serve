#!/bin/bash
uv run -- ray start --head --dashboard-host=0.0.0.0 --include-dashboard=True &

# Wait dynamically until Ray reports
echo "Waiting for Ray to start..."
until uv run -- ray status 2>/dev/null | grep -q "Active"; do
    sleep 2
done
echo "Ray is ready."

uv run -- serve run config.yaml