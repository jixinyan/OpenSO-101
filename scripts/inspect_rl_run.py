import argparse
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
events = EventAccumulator(str(args.run), size_guidance={"scalars": 0}).Reload()
scalars = {name: [{"step": item.step, "value": item.value} for item in events.Scalars(name)]
           for name in events.Tags()["scalars"]}
models = {}
for path in sorted(args.run.glob("model*.pt")):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    models[path.name] = {"iteration": saved["iter"], "parameters": {
        name: {"finite": bool(torch.isfinite(value).all()), "maximum_absolute": float(value.abs().max())}
        for name, value in saved["model_state_dict"].items()}}
report = {"models": models, "scalars": scalars}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(report, stream, indent=2)
for name in ("Loss/learning_rate", "Loss/value_function", "Loss/surrogate", "Policy/mean_noise_std"):
    print(name, scalars[name])
