import json
import os
from concurrent.futures import ThreadPoolExecutor

jsonFiles = [
        #"/afs/crc.nd.edu/user/h/hnelson2/topeft/input_samples/sample_jsons/background_samples/central_UL/UL17_TTJets.json", 
        "/afs/crc.nd.edu/user/h/hnelson2/topeft/input_samples/sample_jsons/background_samples/central_UL/UL17_TTJets_NDSkim.json"
        ]

futures = []

with ThreadPoolExecutor(max_workers=4) as executor:
    for fpath in jsonFiles:
        with open(fpath) as file:
            jobj = json.load(file)

        new_files = []

        for root in jobj["files"]:

            new_dir_path = "/project01/ndcms/hnelson2/mc_samples/central_UL/skims/" 
            new_fpath = root.replace("/store/user/awightma/skims/mc/", "")
            new_fpath = new_fpath.replace("central_bkgd_p1/", "")
            full_path = new_dir_path+new_fpath

            new_files.append(full_path)

            cmd = f"xrdcp root://deepthought.crc.nd.edu/{root} {full_path}"
            print(cmd)

            executor.submit(lambda cmd: os.system(cmd), cmd)
