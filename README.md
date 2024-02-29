# How to install ?

requirements:

```bash
pip install git+https://github.com/SLAMPAI/autoexperiment
pip install clip_benchmark
```

```bash
git clone https://github.com/SLAMPAI/open_clip_scaling
cd open_clip_scaling
git clone https://github.com/mlfoundations/open_clip
```

# How to run ?


First, build sbatch files for each job:

`autoexperiment build config.yaml`


Check the sbatch scripts:

`ls sbatch_scripts/`

Then, run everything:

`autoexperiment run config.yaml`
