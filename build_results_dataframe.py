
import pandas as pd
# import fireducks.pandas as pd

from glob import glob
import re
import sys
import json
import os
import numpy as np
import re
from joblib import Parallel, delayed


def get_samples_per_sec(path):
    all_vals = []
    data = open(path).readlines()
    for line in data:
        if "Train Epoch" in line:
            vals = re.findall("\d+\.\d*\/s,", line)
            vals = [float(v.replace("/s,", "")) for v in vals]
            all_vals.extend(vals)
    return np.mean(all_vals[1:])


def get_params(out_file):
    dic = {}
    for l in open(out_file).readlines():
        index = l.find("| INFO |")
        if index >= 0 and ":" in l:
            l = l[index+len("| INFO |"):]
            l = l.strip()
            try:
                k, v = l.split(":")
                k = k.strip()
                v = v.strip()
                dic[k] = v
            except Exception:
                pass
    return dic

def get_loss(out_file):
    contrasive_losses = []
    caption_losses = []
    with open(out_file, 'r') as file:
        for line in file:
            if 'Train Epoch:' in line and 'Loss:' in line:
                match = re.search(r'Contrastive_loss: (\d+\.\d+) \((\d+\.\d+)\)', line)
                if match:
                    current_loss, average_loss = map(float, match.groups())
                    contrasive_losses.append(average_loss)    
                match = re.search(r'Caption_loss: (\d+\.\d+) \((\d+\.\d+)\)', line)
                if match:
                    current_loss, average_loss = map(float, match.groups())
                    caption_losses.append(average_loss)    

    return {
        "contrastive_loss": contrasive_losses[-1] if len(contrasive_losses) else None,
        "caption_loss": caption_losses[-1] if len(caption_losses) else None
    }

def human(v):
    if v < 10 ** 6:
        return str(v)
    elif v > 10**6 and v < 10**9:
        return (str(v/10**6)+"M").replace(".0M", "M")
    elif v > 10**9:
        return (str(v/10**9)+"B").replace(".0B", "B")


model_profile = pd.concat([
    pd.read_csv("/p/project/laionize/jitsev1_juwelsbooster/open_clip_scaling/model_profile.csv"),
    pd.read_csv("/p/project/laionize/cherti1/open_clip_all_at_once/model_profile_cap.csv"),
])
model_profile_cap_mammut =  pd.read_csv("/p/project/laionize/cherti1/open_clip_all_at_once/model_profile_cap_mammut.csv").set_index("model")
model_profile.to_csv("model_profile.csv", index=False)
model_profile  = model_profile.set_index("model")

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
        params = get_params(out_log)
        model = params["model"]
        samples_per_sec = get_samples_per_sec(out_log)
        losses = get_loss(out_log)
        gpus = int(params["world_size"])
        name = os.path.basename(model_folder)
        if "mammut" in name:
            if "cap_mammut" in path:
                ns = "cap"
            else:
                ns = "mammut"
        elif "coca" in name:
            ns = "coca"
        else:
            ns = "clip"
        mp = model_profile_cap_mammut if ns == "cap_mammut" else model_profile
        dic = {
            'path': path,
            'model': params['model'],
            "pretrain_dataset": params["train_data"],
            "downstream_dataset": data['dataset'],
            'epoch': int(re.search(r"epoch\_([0-9]+).pt", path).groups(1)[0]),
            "total_epochs": int(params['epochs']),
            "name": name,
            "gflops_total":  mp.loc[model].gflops * int(params["epochs"]) * int(params["train_num_samples"]),
            "samples_per_sec": samples_per_sec,
            "samples_per_sec_per_gpu": samples_per_sec / gpus,
            "global_batch_size": int(params["batch_size"]) * gpus,
            "training_time_hours": ((1/samples_per_sec) * int(params["epochs"]) * int(params["train_num_samples"]) ) / 3600,
            "gpus": gpus,
            "total_steps": (int(params["epochs"]) * int(params["train_num_samples"]) ) // ( int(params["batch_size"]) * gpus),
            "task": data["task"],
            "local_batch_size": int(params["batch_size"])
        }
        dic.update(losses)
        dic["namespace"] = ns
        dic["eval_type"] = "log_likelihood" if data["task"].startswith("generative") else "similarity"
        dic["gpu_hours"] = dic["gpus"] * dic["training_time_hours"]
        dic.update(data['metrics'])        
        results.append(dic)
    return (results)
log_folders = [
    "/p/home/jusers/cherti1/juwels/laionize/cherti1/open_clip_scaling/logs/cap_mammut",
    "/p/data1/mmlaion/experiments/autoexp/jjitsev/logs"
]
folders = [folder for log_folder in log_folders for folder in glob(os.path.join(log_folder, "**", "checkpoints"))]
results = Parallel(n_jobs=-1)(delayed(load_results)(f)for f in folders)
rows = [ri for r in results for ri in r]
df = pd.DataFrame(rows)
df['name_epoch'] = df.apply(lambda r:f"{r['name']}{r['epoch']}", axis=1)


rows = []
for n in df.name_epoch.unique():
    sg = df[df.name_epoch==n]
    sg = sg[sg.downstream_dataset.str.startswith("sugar_crepe")]
    if len(sg) == 7:
        rows.append({
            "name": sg.name.iloc[0],
            "epoch": sg.epoch.iloc[0],
            "acc": sg.acc.mean(),
            "gflops_total": sg.gflops_total.mean(),
            "downstream_dataset": "sugar_crepe",
            "namespace": sg.namespace.iloc[0],
            "task": sg.task.iloc[0],
            "model": sg.model.iloc[0],
            "epoch": sg.epoch.iloc[0],
            "total_epochs": sg.total_epochs.iloc[0]
        })
new = pd.DataFrame(rows)
df = pd.concat((df, new))
df["samples_seen_scale_simple"] = df.name.apply(lambda s:s.split("_")[1][1:])
df["samples_seen_scale"] = df["samples_seen_scale_simple"]
df["lr"] = df.name.apply(lambda s: next((float(part[len('lr'):]) for part in s.split('_') if part.startswith('lr')), None))
df["warmup"] = df.name.apply(lambda s: next((int(part.split('_')[1]) if part.startswith('warmup_') else int(part[1:]) for part in s.split('_') if part.startswith('warmup_') or (part.startswith('w') and part[1:].isdigit())), None))
df["model_simple"] = df["model"].apply(lambda s:s.replace("sg_cap_", "").replace("mammut_", "").replace("coca_", ""))
df["name_wo_model"] = df.apply(lambda r:f"{r['lr']}_{r['samples_seen_scale_simple']}_{r['global_batch_size']}_{r['warmup']}", axis=1)
df["namespace_model"] = df.apply(lambda r:f"{r['model']}_{r['namespace']}", axis=1)
df["model_simple_namespace"] = df.apply(lambda r:f"{r['model_simple']}_{r['namespace']}", axis=1)
df["namespace_model_samples_seen_scale"] = df.apply(lambda r:f"{r['model']}_{r['namespace']}_{r['samples_seen_scale']}", axis=1)
df["name_wo_lr"] = df.name.apply(lambda n:"_".join([ni for ni in n.split("_") if "lr" not in ni]))
df["downstream_dataset"] = df.downstream_dataset.apply(lambda s:s.replace("wds/", ""))

df.to_csv("results.csv", index=False)
