#!/usr/bin/env python
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.analysis_tools import PackedSelection

# silence warnings due to using NanoGEN instead of full NanoAOD
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

# Get the lumi for the given year
def get_lumi(year):
    lumi_dict = {
        "2016APV": 19.52,
        "2016": 16.81,
        "2017": 41.48,
        "2018": 59.83
    }
    if year not in lumi_dict.keys():
        raise Exception(f"(ERROR: Unknown year \"{year}\".")
    else:
        return(lumi_dict[year])

# Clean the objects
def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

# Main analysis processor
class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        # Create the histograms with new scikit hist
        self._histo_dict = {
            "jet_flav"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("jet_flav", "pdgID jet flavor", 23, 0, 23)),
            "njets_0p"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets_0p", "njets", 10, 0, 10)), 
            "njets_1p"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets_1p", "njets", 10, 0, 10)),
            "njets"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "njets", 10, 0, 10)),
        }

        # Set the list of hists to to fill
        if hist_lst is None:
            self._hist_lst = list(self._histo_dict.keys())
        else:
            for h in hist_lst:
                if h not in self._histo_dict.keys():
                    raise Exception(f"Error: Cannot specify hist \"{h}\", it is not defined in self._histo_dict")
            self._hist_lst = hist_lst

        print("hist_lst: ", self._hist_lst)


    @property
    def columns(self):
        return self._columns

    def process(self, events):

        # Dataset parameters
        dataset = events.metadata['dataset']
        hist_axis_name = self._samples[dataset]["histAxisName"]

        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights']

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None


        ######## Initialize objects  ########

        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        jets = events.GenJet


        ######## Lep selection  ########

        e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        leps = ak.concatenate([ele[e_selec],mu[m_selec]],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        l0 = leps[ak.argmax(leps.pt, axis=-1, keepdims=True)]


        ######## Jet selection  ########

        jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        jets_clean = jets[is_clean(jets, leps, drmin=0.4)]
        ht = ak.sum(jets_clean.pt, axis=-1)
        j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]


        ######## Top selection ########

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
        mtt = (gen_top[:,0] + gen_top[:,1]).mass
        t0 = gen_top[ak.argmax(gen_top.pt, axis=-1, keepdims=True)]


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


        ######## Calculate Variables ########

        lhe_htincoming = events.LHE.HTIncoming
        lhe_ht = events.LHE.HT

        j0pt = j0.pt
        jets_pt = jets_clean.sum()

        l0pt = l0.pt
        leps_cut = leps[event_selection_mask]
        dr_cut = leps_cut[:,0].delta_r(leps_cut[:,1])
        mll = (leps_cut[:,0] + leps_cut[:,1]).mass

        tops_pt = gen_top.sum().pt
        avg_top_pt = np.divide(tops_pt, ntops)
        t0eta = t0.eta 
        t0phi = t0.phi


        ######## Normalization ########

        # Normalize by (xsec/sow)
        #lumi = 1000.0*get_lumi(year)
        # norm = (xsec/sow)
        # norm = (1/sow)
        # norm = (1/200)

        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])

        event_weights = genw

        eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None


        njets_cut = ak.num(jets_clean[event_selection_mask])


        # Jet Flavor
        jet_flav = abs(jets_clean[event_selection_mask].partonFlavour)
        jet_flav_eft = np.repeat(eft_coeffs[event_selection_mask], njets_cut, axis=0)
        jet_flav_weight = np.repeat(event_weights[event_selection_mask], njets_cut, axis=0)

        nMEPartons = events.Generator.nMEPartonsFiltered[event_selection_mask]
        njets_0p = njets_cut[nMEPartons == 0]
        njets_1p = njets_cut[nMEPartons == 1]

        ######## Fill histos ########

        hout = self._histo_dict

        jet_flav_fill_info = {
            "jet_flav"  : ak.flatten(jet_flav), 
            "sample"    : hist_axis_name, 
            "weight"    : jet_flav_weight,
            "eft_coeff" : jet_flav_eft,
        }

        njets_fill_info = {
            "njets"     : njets_cut,
            "sample"    : hist_axis_name,
            "weight"    : event_weights[event_selection_mask],
            "eft_coeff" : eft_coeffs_cut,
        }

        njets_0p_fill_info = {
            "njets_0p"  : njets_0p,
            "sample"    : hist_axis_name,
            "weight"    : event_weights[event_selection_mask][nMEPartons == 0],
            "eft_coeff" : eft_coeffs_cut[nMEPartons == 0],
        }

        njets_1p_fill_info = {
            "njets_1p"  : njets_1p,
            "sample"    : hist_axis_name,
            "weight"    : event_weights[event_selection_mask][nMEPartons == 1],
            "eft_coeff" : eft_coeffs_cut[nMEPartons == 1],
        }

        hout["jet_flav"].fill(**jet_flav_fill_info)
        hout["njets"].fill(**njets_fill_info)
        hout["njets_0p"].fill(**njets_0p_fill_info)
        hout["njets_1p"].fill(**njets_1p_fill_info)

        
        return hout


    def postprocess(self, accumulator):
        return accumulator
