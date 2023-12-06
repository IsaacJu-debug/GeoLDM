#!/bin/bash

# List of job IDs to cancel
job_ids=(
    37291893
    37291894
    37291895
    37291896
    37291897
    37291898
    37291899
    37291900
    37291903
    37291904
    37292831
    37292832
    37292833
    37292834
    37292835
    37292836
    37292837
    37292838
    37292839
    37292840
    37292841
    37292842
    37292871
    37292872
    37291892
    37291891
    37291889
)

# Loop through each job ID and cancel it
for job_id in "${job_ids[@]}"; do
    echo "Cancelling job: $job_id"
    scancel $job_id
done

echo "All specified jobs have been cancelled."

