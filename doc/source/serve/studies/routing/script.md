```

# single
python scripts/run_experiments.py --run-type full --warmup 30 --duration 60 --load-level 1.0 --algorithm pow2 --scale xlarge --s3-bucket abrar-test-bucket-123 --clear-s3 | tee /tmp/experiment.log

# Small group
python scripts/run_experiments.py --run-type full --warmup 60 --duration 60 --scale large --s3-bucket abrar-test-bucket-123 --clear-s3 | tee /tmp/experiment.log

# Aggregate data
python -m analysis.aggregate s3://abrar-test-bucket-123/routing-study/20260110_071843/  --output /tmp/routing_results/metrics.json     --csv-output /tmp/routing_results/summary.csv

# Generate visualizations
python -m analysis.visualize --input /tmp/routing_results/summary.csv --output-dir figures

# Generate report
python -m analysis.report --input /tmp/routing_results/summary.csv --output /tmp/routing_results/RESULTS.md
```