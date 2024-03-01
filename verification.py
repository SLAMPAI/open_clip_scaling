from autoexperiment.template import generate_job_defs
jobdefs = generate_job_defs("config.yaml")
for jobdef in jobdefs:
    samples_seen = (jobdef.params["samples_seen_scale"])
    if samples_seen.endswith("M"):
        samples_seen = float(samples_seen[:-1]) * 10**6
    elif samples_seen.endswith("K"):
        samples_seen = float(samples_seen[:-1]) * 10**3
    elif samples_seen.endswith("B"):
        samples_seen = float(samples_seen[:-1]) * 10**9
    assert jobdef.params["train_num_samples"] * jobdef.params["epochs"] == samples_seen
print("Successful")