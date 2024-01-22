import json
import os
from concurrent.futures import ThreadPoolExecutor

jsonFiles = [
        "/afs/crc.nd.edu/user/h/hnelson2/ttbarEFT/input_samples/sample_jsons/central_UL/UL17_TTJets_NDSkim.json"
        ]


for fpath in jsonFiles:
    with open(fpath) as file:
        jobj = json.load(file)

    flist = []
    for root in jobj["files"]:
        new_fpath = root.replace("/store/user/awightma/skims/mc/new-lepMVA-v2/central_bkgd_p1", "/project01/ndcms/hnelson2/mc_samples/central_UL/skims/new-lepMVA-v2")
        flist.append(new_fpath) 

    jobj["files"] = flist

    with open("/afs/crc.nd.edu/user/h/hnelson2/ttbarEFT/input_samples/sample_jsons/central_UL/new_UL17_TTJets_NDSkim.json","w") as f2:
        json.dump(jobj, f2, indent=4)

