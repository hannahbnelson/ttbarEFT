import os
from pathlib import Path
import json
import argparse
import subprocess
import shlex

"""
Example:
python3 get_dataset_from_das.py datasets.txt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options parser")
    parser.add_argument("dataset_file", help="Dataset file to load.", type=str)
    parser.add_argument("-of", "--outputFile", help="Output JSON for datasets.", type=str, default=None, required=False)
    args = parser.parse_args()

    ## Output file setup
    if args.outputFile is None:
        args.outputFile = "tmp_dataset.json"

    print(f"Using output filename: {args.outputFile}")
    if os.path.exists(args.outputFile):
        assert not os.path.exists(args.outputFile), "Output file will not be overwritten! Backup the file and try again."

    ## Get datasets from file
    dbs_out = []
    datasets = []
    with open(args.dataset_file, "r") as f:
        for line in f:
            datasets.append(line.strip())

    ## Get files in datasets from DAS using dasgoclient
    for dataset in datasets:
        dbs_out.extend(
            subprocess.run(
                shlex.split(f'dasgoclient -query "file dataset={dataset}"'),
                stdout = subprocess.PIPE,
                universal_newlines = True).stdout.split("\n")[:-1]
        )

    print(dbs_out)

    # ## Fill dictionary with file paths
    # d = {}
    # previous_dataset = ""
    # current_dataset = ""
    # redirector = "root://cmsxrootd.fnal.gov//"
    # for line in dbs_out:
    #     current_dataset = line.split("/")[1]

    #     if current_dataset.lower() != previous_dataset.lower():
    #         d.update({current_dataset: list().copy()})

    #     # d[current_dataset].append(f"{redirector}{line}")    
    #     d[current_dataset].append(f"{line}")
    #     previous_dataset = current_dataset

    # ## Dump datasets/files dict to JSON
    # with open(args.outputFile, "w") as f:
    #     json.dump(
    #         d,
    #         f,
    #         indent = 4,
    #         sort_keys=True
    #     )
    #     print(f"Dumped JSON to {args.outputFile}!")