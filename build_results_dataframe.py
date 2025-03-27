
import pandas as pd
from glob import glob
import re
import json
import os
import numpy as np
import re
from joblib import Parallel, delayed
def samples_seen_from_str(samples):
    if samples.endswith("M"):
        samples = float(samples[:-1]) * 10**6
    elif samples.endswith("B"):
        samples = float(samples[:-1]) * 10**9
    elif samples.endswith("K"):
        samples = float(samples[:-1]) * 10**3
    return samples
def parse_out_log(path):
    params = {}
    all_vals = []
    contrastive_losses = []
    caption_losses = []
    
    for line in open(path).readlines():
        # Extract parameters
        index = line.find("| INFO |")
        if index >= 0 and ":" in line:
            l = line[index + len("| INFO |"):].strip()
            try:
                k, v = l.split(":", 1)
                params[k.strip()] = v.strip()
            except ValueError:
                continue  # Skip lines that don't conform
        
        # Extract samples per second
        if "Train Epoch" in line:
            vals = re.findall(r"([\d\.]+)\/s,", line)
            all_vals.extend([float(v) for v in vals])
        
        # Extract losses
        if 'Train Epoch:' in line and 'Loss:' in line:
            con_loss_match = re.search(r'Contrastive_loss: [\d\.]+ \(([\d\.]+)\)', line)
            cap_loss_match = re.search(r'Caption_loss: [\d\.]+ \(([\d\.]+)\)', line)
            if con_loss_match:
                contrastive_losses.append(float(con_loss_match.group(1)))
            if cap_loss_match:
                caption_losses.append(float(cap_loss_match.group(1)))
    
    samples_per_sec = np.mean(all_vals[1:]) if len(all_vals) > 1 else (all_vals[0] if all_vals else None)
    losses = {
        "train_contrastive_loss": contrastive_losses[-1] if contrastive_losses else None,
        "train_caption_loss": caption_losses[-1] if caption_losses else None
    }
    return params, samples_per_sec, losses


model_profile = pd.concat([
    pd.read_csv("/p/project/laionize/jitsev1_juwelsbooster/open_clip_scaling/model_profile.csv"),
    pd.read_csv("/p/project/laionize/cherti1/open_clip_all_at_once/model_profile_cap.csv"),
])
model_profile_cap_mammut =  pd.read_csv("/p/project/laionize/cherti1/open_clip_all_at_once/model_profile_cap_mammut.csv").set_index("model")
model_profile.to_csv("model_profile.csv", index=False)
model_profile  = model_profile.set_index("model")
def get_pretrain_dataset(train_data):
    return {
        '/p/data1/mmlaion/datacomp/datacomp_1B/flat/{0000000..0139827}.tar': "datacomp_1b",
       '/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{0000000..0139827}.tar': "datacomp_1b",
       '/p/scratch/laionize/cherti1/datacomp_1B_recap/{0000000..0139827}.tar': "datacomp_1b_recap",
    }[train_data]
    
    
def load_results(folder):

    paths = glob(os.path.join(folder, "*.json"))
    results = []
    for path in paths:
        if 'latest' in path:
            continue
        data = json.load(open(path))
        
        
        model_folder = os.path.dirname(os.path.dirname(path))
        out_log = os.path.join(model_folder, "out.log")
        
        if not os.path.exists(out_log):
            continue
        params, samples_per_sec, losses = parse_out_log(out_log)
        model = params["model"]
        gpus = int(params["world_size"])
        name = os.path.basename(model_folder)


        if model.startswith("mammut_"):
            cw = float(params["coca_contrastive_loss_weight"])
            ns = "cap" if cw == 0 else "mammut"
        elif model.startswith("coca_"):
            ns = "coca"
        else:
            ns = "clip"

        mp = model_profile_cap_mammut if ns == "cap_mammut" else model_profile
        epoch = int(re.search(r"epoch\_([0-9]+).pt", path).groups(1)[0])
        dic = {
            "name": name,
            'path': path,
            "model_folder": model_folder,
            'model': params['model'],
            "pretrain_dataset": get_pretrain_dataset(params["train_data"]),
            "downstream_dataset": data['dataset'],
            'epoch': epoch,
            "samples_seen": int(params["train_num_samples"]) * epoch,
            "steps":  (epoch) * int(params["train_num_samples"]) // ( int(params["batch_size"]) * gpus),
            "samples_seen_per_epoch": int(params["train_num_samples"]),                        
            "gflops": mp.loc[model].gflops * epoch * int(params["train_num_samples"]),            
            "training_time_hours": ((1/samples_per_sec) * epoch * int(params["train_num_samples"]) ) / 3600,            
            "total_epochs": int(params['epochs']),
            "samples_per_sec": samples_per_sec,
            "samples_per_sec_per_gpu": samples_per_sec / gpus,
            "global_batch_size": int(params["batch_size"]) * gpus,            
            "gpus": gpus,
            "task": data["task"],
            "local_batch_size": int(params["batch_size"]),
            "warmup": int(params["warmup"]),
            "lr": float(params["lr"]),
            "lr_scheduler": params["lr_scheduler"],
        }
        dic.update(losses)
        dic["namespace"] = ns
        dic["eval_type"] = "log_likelihood" if data["task"].startswith("generative") else "similarity"
        dic["gpu_hours"] = dic["gpus"] * dic["training_time_hours"]
        dic.update(data['metrics'])        
        results.append(dic)
    return (results)
log_folders = [
    "/p/data1/mmlaion/cherti1/open_clip_scaling/logs/cap_mammut",
    "/p/data1/mmlaion/experiments/autoexp/jjitsev/logs",
    "/p/data1/mmlaion/cherti1/open_clip_scaling/logs/const_lr",
    "/p/data1/mmlaion/porian1/mammut_clip_const"
]
folders = [
    folder for log_folder in log_folders 
    for folder in glob(os.path.join(log_folder, "**", "checkpoints"), recursive=True)
]

results = Parallel(n_jobs=-1)(delayed(load_results)(f)for f in folders)
rows = [ri for r in results for ri in r]
df = pd.DataFrame(rows)
df["name_epoch"] = df["name"] + "_" + df["epoch"].astype(str)
df["model_folder_epoch"] = df.model_folder + "_epoch_" + df.epoch.astype(str)
print("Done loading results")

# covers the case where same downream dataset has been evaluated several times with different filenames
df = df.drop_duplicates(subset=["model_folder_epoch", "downstream_dataset"]) 

suites = {
    "datacomp_classification": {
        "metric": "acc1",
        "datasets":[
            "imagenet1k",
            "wds/wds_wilds-fmow_test",
            "wds/wds_vtab-pcam_test",
            "wds/wds_geode_test",
            "wds/wds_voc2007_test",
            "wds/wds_renderedsst2_test",
            "wds/wds_gtsrb_test",
            "wds/wds_food101_test",
            "wds/wds_imagenet-a_test",
            "wds/wds_fgvc_aircraft_test",
            "wds/wds_imagenet-r_test",
            "wds/wds_wilds-iwildcam_test",
            "wds/wds_vtab-flowers_test",
            "wds/wds_country211_test",
            "wds/wds_vtab-cifar100_test",
            "wds/wds_mnist_test",
            "wds/wds_sun397_test",
            "wds/wds_imagenetv2_test",
            "wds/wds_stl10_test",
            "wds/wds_cars_test",
            "wds/wds_objectnet_test",
            "wds/wds_vtab-clevr_count_all_test",
            "wds/wds_vtab-kitti_closest_vehicle_distance_test",
            "wds/wds_cifar10_test",
            "wds/wds_dollar_street_test",
            "wds/wds_vtab-dtd_test",
            "wds/wds_imagenet-o_test",
            "wds/wds_vtab-resisc45_test",
            "wds/wds_imagenet_sketch_test",
            "wds/wds_vtab-eurosat_test",
            "wds/wds_wilds-camelyon17_test",
            "wds/wds_vtab-caltech101_test",
            "wds/wds_vtab-svhn_test",
            "wds/wds_vtab-clevr_closest_object_distance_test",
            "wds/wds_vtab-pets_test"
        ],
    },
    "sugar_crepe": {
        "metric": "acc",
        "datasets":[
            "sugar_crepe/swap_obj",
            "sugar_crepe/swap_att",
            "sugar_crepe/add_att",
            "sugar_crepe/add_obj",
            "sugar_crepe/replace_att",
            "sugar_crepe/replace_obj",
            "sugar_crepe/replace_rel"
        ],
    }
}
new_dfs = []
for suite_name, suite in suites.items():
    datasets = suite["datasets"]
    metric = suite["metric"]
    d = df[df.downstream_dataset.isin(datasets)]
    d = d.groupby("model_folder_epoch").filter(lambda group: len(group) == len(datasets))
    if len(d):
        aggs = {}
        for col in set(df.columns) - set(["model_folder_epoch"]):
            aggs[col] = "first"
        aggs[metric] = "mean"
        d = d.groupby("model_folder_epoch").agg(aggs)
        d = d.reset_index()
        d["downstream_dataset"] = suite_name
        new_dfs.append(d)
df = pd.concat([df] + new_dfs)

df["model_simple"] = df["model"].apply(lambda s:s.replace("sg_cap_", "").replace("mammut_", "").replace("coca_", ""))
df["model_simple_namespace"] = df.apply(lambda r:f"{r['model_simple']}_{r['namespace']}", axis=1)
df["downstream_dataset"] = df.downstream_dataset.apply(lambda s:s.replace("wds/", ""))

scales = [
    "1.28M",
    "3M",
    "6.4M",
    "12.8M",
    "30M",
    "64M",
    "128M",
    "300M",
    "640M",
    "1.28B",
    "3B",
    "12.8B",
]
scales_numeric = [samples_seen_from_str(s) for s in scales]
def human(v):
    # check closest from `scales`
    dist = np.abs(np.array(scales_numeric) - v)
    if dist.min() < 1e9:
        idx = dist.argmin()
        return scales[idx]
    else:
        if v < 10 ** 6:
            return str(v)
        elif v > 10**6 and v < 10**9:
            return (str(v/10**6)+"M").replace(".0M", "M")
        elif v > 10**9:
            return (str(v/10**9)+"B").replace(".0B", "B")

df["samples_seen_scale_pretty"] = df.samples_seen.apply(lambda s:human(s))
df.to_csv("results.csv", index=False)
