#!/usr/bin/env python
import awkward as ak
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from coffea import processor
from coffea.analysis_tools import PackedSelection
from topcoffea.modules import utils
import topcoffea.modules.eft_helper as efth

SM_pts = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

rwgt_pt1 = {"ctGIm": 1.0, "ctGRe":1.0, "cQj38": 3.0, "cQj18": 3.0,
          "cQu8": 3.0, "cQd8": 3.0, "ctj8": 3.0, "ctu8": 3.0,
          "ctd8": 3.0, "cQj31": 3.0, "cQj11": 3.0, "cQu1": 3.0,
          "cQd1": 3.0, "ctj1": 3.0, "ctu1": 3.0, "ctd1": 3.0}

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

# Create list of wc values in the correct order
def order_wc_values(wcs, ref_pts):
    '''Returns list of wc values in the same order as the list of wc names based on a dictionary
    '''
    wc_names = wcs
    ref_pts = ref_pts

    wc_values = []
    for i in wc_names:
        wc_values.append(ref_pts[i])

    return wc_values

# Calculate event weights from wc values and eft fit coefficients
def calc_event_weights(eft_coeffs, wc_vals):
    '''Returns an array that contains the event weight for each event.
    eft_coeffs: Array of eft fit coefficients for each event
    wc_vals: wilson coefficient values desired for the event weight calculation, listed in the same order as the wc_lst
             such that the multiplication with eft_coeffs is correct
             The correct ordering can be achieved with the order_wc_values function
    '''
    event_weight = np.empty_like(eft_coeffs)

    wcs = np.hstack((np.ones(1),wc_vals))
    wc_cross_terms = []
    index = 0
    for j in range(len(wcs)):
        for k in range (j+1):
            term = wcs[j]*wcs[k]
            wc_cross_terms.append(term)
    event_weight = np.sum(np.multiply(wc_cross_terms, eft_coeffs), axis=1)

    return event_weight

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        print("self._samples", self._samples)
        print("self._wc_names_lst", self._wc_names_lst)

        # Create the histograms with new scikit hist
        self._histo_dict = {
            "avg_top_pt"   		: Hist(hist.axis.Regular(bins=40, start=0, stop=400, name="average top $p_T$ [GeV]"), storage="weight"),
            "l0pt"         		: Hist(hist.axis.Regular(bins=40, start=0, stop=400, name="leading lepton $p_T$ [GeV]"), storage="weight"),
            "dr_leps"      		: Hist(hist.axis.Regular(bins=30, start=0, stop=6, name="$\Delta R$ (leading lepton, subleading lepton)"), storage="weight"),
            "mtt"          		: Hist(hist.axis.Regular(bins=60, start=0, stop=1200, name="invariant mass of tops"), storage="weight"),
            "njets"        		: Hist(hist.axis.Regular(bins=10, start=0, stop=10, name="njets"), storage="weight"),
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
        # sow_before_selec = self._samples[dataset]['nSumOfWeightsBeforeSelec']
        # sow_after_selec = self._samples[dataset]['nSumOfWeightsAfterSelec']

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None


        # Initialize objects
        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        nu = ak.concatenate([nu_ele,nu_mu],axis=1)
        jets = events.GenJet

        ######## Lep selection  ########

        e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        leps = ak.concatenate([ele[e_selec],mu[m_selec]],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        l0 = leps[ak.argmax(leps.pt, axis=-1, keepdims=True)]

        ######## Jet selection  ########

        jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
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
        dr_l0 = leps_cut[:,0]
        dr_l1 = leps_cut[:,1]
        l0pt_cut = l0.pt[event_selection_mask]
        dr_cut = dr_l0.delta_r(dr_l1)

        tops_pt_cut = gen_top.sum().pt[event_selection_mask]
        avg_top_pt_cut = np.divide(tops_pt_cut, 2.0)

        njets_cut = njets[event_selection_mask]
        nleps_cut = nleps[event_selection_mask]
        mtt_cut = mtt[event_selection_mask]
        ht_cut = ht[event_selection_mask]
        ntops_cut = ntops[event_selection_mask]
        jets_pt_cut = jets_clean.sum().pt[event_selection_mask]
        j0pt_cut = j0.pt[event_selection_mask]
        mll = (leps_cut[:,0] + leps_cut[:,1]).mass
        lhe_ht = events.LHE.HT[event_selection_mask]
        lhe_htincoming = events.LHE.HTIncoming[event_selection_mask] 

        eft_coeffs_cut = eft_coeffs[event_selection_mask]

        wc_lst_SM = order_wc_values(self._wc_names_lst, SM_pts)
        event_weights_SM = calc_event_weights(eft_coeffs_cut, wc_lst_SM)

        wc_lst_pt1 = order_wc_values(self._wc_names_lst, rwgt_pt1)
        event_weights_pt1 = calc_event_weights(eft_coeffs_cut, wc_lst_pt1)

        ######## Normalization ########

        # Normalize by (xsec/sow)
        #lumi = 1000.0*get_lumi(year)
        norm = (xsec/sow)
        # norm = (1/sow)
        # norm = 1.0

        # w2 = np.square(event_weights_SM*norm)

        ######## Fill histos ########
        hout = self._histo_dict
        variables_to_fill = {
            "avg_top_pt"	: avg_top_pt_cut,
            "l0pt"			: ak.flatten(l0pt_cut),
            "dr_leps"		: dr_cut,
            "mtt"			: mtt_cut,
            "njets"			: njets_cut, 
        }

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            # hout[var_name].fill(var_values, weight=event_weights_SM*norm)
            hout[var_name].fill(var_values, weight=event_weights_pt1*norm)
            # hout[var_name].fill(var_values, weight=w2)

        return hout

    def postprocess(self, accumulator):
        return accumulator

