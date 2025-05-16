import awkward as ak
import numpy as np
#import hist
from hist import Hist
from coffea import hist
from topcoffea.modules.HistEFT import HistEFT
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from coffea.analysis_tools import PackedSelection
from topcoffea.modules import utils
import topcoffea.modules.eft_helper as efth

sm_pts = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

# Clean the objects
def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

files = [
   "/project01/ndcms/hnelson2/ttbarEFT/nanoGen/TT1j2l_cQj31/nanoGen_101.root",
   "/project01/ndcms/hnelson2/mc_samples/central_UL/skims/new-lepMVA-v2/FullRun2/v2/UL17_TTJets/output_3383.root"
]

for fname in files: 
    events = NanoEventsFactory.from_root(
        fname,
        schemaclass = NanoAODSchema.v6,
        metadata = {"dataset": "TT1j2l"},
    ).events()

    print("\n File: ", fname, "\n")

    wc_lst = utils.get_list_of_wc_names(fname)
    print("wc list: ", wc_lst)

    dataset = events.metadata['dataset']

    eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None

    if eft_coeffs is None:
        event_weights = events["genWeight"]
    else:
        event_weights = np.ones_like(events['event'])

    # Initialize objects
    genpart = events.GenPart
    is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
    ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
    mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
    jets = events.GenJet

    ######## Lep selection  ########

    e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
    m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
    leps = ak.concatenate([ele[e_selec],mu[m_selec]],axis=1)

    ######## Jet selection  ########

    jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
    jets_clean = jets[is_clean(jets, leps, drmin=0.4)]
    ht = ak.sum(jets_clean.pt, axis=-1)
    j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]

    ######## Top selection ########

    gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
    mtt = (gen_top[:,0] + gen_top[:,1]).mass

    ######## Event selections ########

    nleps = ak.num(leps)
    njets = ak.num(jets_clean)
    ntops = ak.num(gen_top)

    at_least_two_leps = ak.fill_none(nleps>=2,False)
    at_least_two_jets = ak.fill_none(njets>=2, False)

    selections = PackedSelection()
    selections.add('2l', at_least_two_leps)
    selections.add('2j', at_least_two_jets)
    event_selection_mask = selections.all('2l', '2j')

    leps_cut = leps[event_selection_mask]
    tops_pt_cut = gen_top.sum().pt[event_selection_mask]
    njets_cut = njets[event_selection_mask]
    nleps_cut = nleps[event_selection_mask]
    mtt_cut = mtt[event_selection_mask]
    ht_cut = ht[event_selection_mask]
    ntops_cut = ntops[event_selection_mask]
    jets_pt_cut = jets_clean.sum().pt[event_selection_mask]
    j0pt_cut = j0.pt[event_selection_mask]
    mll = (leps_cut[:,0] + leps_cut[:,1]).mass

    eft_coeffs_cut = eft_coeffs
    if eft_coeffs is None: 
        genw = events["genWeight"]
    else:
        genw = np.ones_like(events['event'])
    
    event_weights = genw

    fill_info = {
        "njets" : njets_cut,
        "sample" : "njets", 
        "weight" : event_weights[event_selection_mask], 
        "eft_coeff" : eft_coeffs_cut,
    }

    h = HistEFT("Events", wc_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "njets", 10, 0, 10))
    h.fill(**fill_info)

    print(h)