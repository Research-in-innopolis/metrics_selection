import numpy
import pandas as pd
import os
import torch
from numpy import float64
import json
from tqdm.auto import trange
import sys

from ACO import ACO

# Define metrics for classes and methods
class_metrics = [
    'CBO', 'CLOC', 'DIT', 'LCOM5', 'LLOC', 'LOC', 'NLM', 'NM', 'NOA', 'NOC',
    'NOD', 'NOI', 'NOP', 'NOS', 'RFC', 'TLLOC', 'TLOC', 'TNLM', 'TNM',
    'TNOS', 'WMC'
]

method_metrics = [
    'CLOC', 'DLOC', 'LLOC', 'LOC', 'McCC', 'NOI', 'NOS', 'NUMPAR', 'TCLOC',
    'TLLOC', 'TLOC', 'TNOS'
]

# Constants for ACO
ACO_COUNTS = 30
ACO_ITERATIONS = 30


def main():
    useCuda = sys.argv[1] == "True"

    device = torch.device('cpu')

    if useCuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    methods_dict = {}
    class_dict = {}

    # Walk through the data directory
    for address, dirs, files in os.walk('data'):
        for name in files:
            path = os.path.join(address, name)
            is_class_file = path.endswith("-Class.csv")
            key = address.replace("data\\", "")
            if is_class_file and os.path.exists("res2/" + key + ".json"):
                print("Skip: " + key + " CL: ", is_class_file)
                continue
            if not is_class_file and os.path.exists("res2/" + key + "_m.json"):
                print("Skip: " + key + " CL: ", is_class_file)
                continue

            print(f"Processing: {key}. Is Class: {is_class_file}")

            class_dict[key] = []
            methods_dict[key] = []
            df = pd.read_csv(path)

            if is_class_file:
                df = df[class_metrics]
                data = torch.from_numpy(df.to_numpy(dtype=float64)).to(device)
                n_features = data.shape[1]

                for i in trange(2, n_features):
                    aco = ACO(n_ants=ACO_COUNTS, n_features=n_features, n_selected_features=i, max_iter=ACO_ITERATIONS,
                              device=device)
                    selected_metrics, cost = aco.run(data, use_tqdm=True, batch_size=9000)
                    class_dict[key].append({
                        "selected_metrics": df.columns[selected_metrics.cpu()].tolist(),
                        "sammon_error": float(cost.cpu())
                    })

                with open(f"res2/{key}.json", "w") as f_metrics:
                    json.dump(class_dict, f_metrics, indent=4)
            else:
                df = df[method_metrics]
                np_data = df.to_numpy(dtype=float64)
                np_data[numpy.isnan(np_data)] = 0
                np_data[numpy.isinf(np_data)] = 0
                np_data[np_data < 0] = 0
                data = torch.from_numpy(np_data).to(device)
                n_features = data.shape[1]

                for i in trange(2, n_features):
                    aco = ACO(n_ants=ACO_COUNTS, n_features=n_features, n_selected_features=i, max_iter=ACO_ITERATIONS,
                              device=device)
                    selected_metrics, cost = aco.run(data, use_tqdm=True, batch_size=9000)
                    methods_dict[key].append({
                        "selected_metrics": df.columns[selected_metrics.cpu()].tolist(),
                        "sammon_error": float(cost.cpu())
                    })

                with open(f"res2/{key}_m.json", "w") as f_metrics:
                    json.dump(methods_dict, f_metrics, indent=4)


if __name__ == '__main__':
    main()
