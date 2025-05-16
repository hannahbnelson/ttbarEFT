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

SM_pt = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

st_pt = {"ctGIm": -0.5, "ctGRe":-0.5, "cQj38": 1.5, "cQj18": 1.5,
          "cQu8": 1.5, "cQd8": 1.5, "ctj8": 1.5, "ctu8": 1.5,
          "ctd8": 1.5, "cQj31": 1.5, "cQj11": 1.5, "cQu1": 1.5,
          "cQd1": 1.5, "ctj1": 1.5, "ctu1": 1.5, "ctd1": 1.5}

rwgt_pt1 = {"ctGIm": 1.0, "ctGRe":1.0, "cQj38": 3.0, "cQj18": 3.0,
          "cQu8": 3.0, "cQd8": 3.0, "ctj8": 3.0, "ctu8": 3.0,
          "ctd8": 3.0, "cQj31": 3.0, "cQj11": 3.0, "cQu1": 3.0,
          "cQd1": 3.0, "ctj1": 3.0, "ctu1": 3.0, "ctd1": 3.0}

# Close to the limits we have, positive ctG
rwgt_pt2 = {'ctGIm': 1.0, 'ctGRe':1.0, 'cQj38':6.0, 'cQj18':5.0,
            'cQu8':4.0, 'cQd8':9.0, 'ctj8':3.0, 'ctu8':4.5,
            'ctd8':9.0, 'cQj31':3.0, 'cQj11':3.0, 'cQu1':3.0,
            'cQd1':4.5, 'ctj1':2.5, 'ctu1':3.2, 'ctd1':4.5}

# Larger WC values (roughly double limits we have)
rwgt_pt3 = {'ctGIm': 3, 'ctGRe':3, 'cQj38':12.0, 'cQj18':10.0,
            'cQu8':8.0, 'cQd8':18.0, 'ctj8':6.0, 'ctu8':9,
            'ctd8':18.0, 'cQj31':6.0, 'cQj11':6.0, 'cQu1':6.0,
            'cQd1':9, 'ctj1':5, 'ctu1':7, 'ctd1':9}

# Larger WC values, ctG slightly smaller in commparison
rwgt_pt4 = {'ctGIm': 1.5, 'ctGRe':1.5, 'cQj38':12.0, 'cQj18':10.0,
            'cQu8':8.0, 'cQd8':18.0, 'ctj8':6.0, 'ctu8':9,
            'ctd8':18.0, 'cQj31':6.0, 'cQj11':6.0, 'cQu1':6.0,
            'cQd1':9, 'ctj1':5, 'ctu1':7, 'ctd1':9}

# Really large WC values
rwgt_pt5 = {'ctGIm': 4, 'ctGRe':4, 'cQj38':12.0, 'cQj18':12.0,
            'cQu8':12.0, 'cQd8':12.0, 'ctj8':12.0, 'ctu8':12,
            'ctd8':12.0, 'cQj31':12.0, 'cQj11':12.0, 'cQu1':12.0,
            'cQd1':12, 'ctj1':12, 'ctu1':12, 'ctd1':12}

rwgt_choice = SM_pt
# rwgt_choice = st_pt

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
            "njets"             : Hist(hist.axis.Regular(bins=10, start=0, stop=10, name="njets", label='njets'), storage="weight"),
            "nleps"             : Hist(hist.axis.Regular(bins=10, start=0, stop=10, name="nleps", label='nleps'), storage="weight"),
            "ntops"             : Hist(hist.axis.Regular(bins=10, start=0, stop=10, name="ntops", label='ntops'), storage="weight"),
            "mtt"               : Hist(hist.axis.Regular(bins=30, start=0, stop=1500, name='mtt', label='GEN invariant mass of tops [GeV]'), storage="weight"),
            "lhe_mtt"           : Hist(hist.axis.Regular(bins=30, start=0, stop=1500, name='lhe_mtt', label='LHE invariant mass of tops [GeV]'), storage="weight"),
            "mll"               : Hist(hist.axis.Regular(bins=16, start=0, stop=800, name='mll', label='invariant mass of leptons [GeV]'), storage="weight"),
            "dr_leps"           : Hist(hist.axis.Regular(bins=24, start=0, stop=6, name="$\Delta R$ (leading lepton, subleading lepton)"), storage="weight"),
            "l0pt"         		: Hist(hist.axis.Regular(bins=20, start=0, stop=400, name="leading lepton $p_T$ [GeV]"), storage="weight"),
            "tops_pt"           : Hist(hist.axis.Regular(bins=35, start=0, stop=700, name="tops_pt", label="$p_T$ of the sum of the tops [GeV]"), storage="weight"),
            "avg_top_pt"        : Hist(hist.axis.Regular(bins=20, start=0, stop=400, name="avg_top_pt", label="average top $p_T$ [GeV]"), storage="weight"),
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

        lhepart = events.LHEPart
        lhe_mtt = (lhepart[:,0]+lhepart[:,1]+lhepart[:,2]+lhepart[:,3]+lhepart[:,4]+lhepart[:,5]+lhepart[:,6]+lhepart[:,7]).mass

        ######## Event selections ########

        nleps = ak.num(leps)
        njets = ak.num(jets_clean)
        ntops = ak.num(gen_top)

        # Standard 2l2j selections
        at_least_two_leps = ak.fill_none(nleps>=2,False)
        at_least_two_jets = ak.fill_none(njets>=2,False)

        selections = PackedSelection()
        selections.add('2l', at_least_two_leps)
        selections.add('2j', at_least_two_jets)
        event_selection_mask = selections.all('2l', '2j')

        ######## Variables with Selections ########
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

        lhe_mtt_cut = lhe_mtt[event_selection_mask]

        eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None

        if eft_coeffs is None:
            genw = events["genWeight"]
            event_weights = genw[event_selection_mask]

        else: 
            wc_lst = order_wc_values(self._wc_names_lst, rwgt_choice)
            event_weights = calc_event_weights(eft_coeffs_cut, wc_lst)

        ######## Normalization ########

        # Normalize by (xsec/sow)
        #lumi = 1000.0*get_lumi(year)
        norm = (xsec/sow)
        # norm = 1.0

        # w2 = np.square(event_weights_SM*norm)
            
        counts = np.ones_like(events['event'])[event_selection_mask]


        ######## Fill histos ########
        hout = self._histo_dict
        variables_to_fill = {
            "njets"     : njets_cut,
            "nleps"     : nleps_cut,
            "ntops"     : ntops_cut,
            "mtt"       : mtt_cut,
            "lhe_mtt"   : lhe_mtt_cut,
            "mll"       : mll,
            "dr_leps"   : dr_cut,
            "l0pt"      : ak.flatten(l0pt_cut),
            "tops_pt"   : tops_pt_cut,
            "avg_top_pt": avg_top_pt_cut,
            "lhe_mtt"   : lhe_mtt_cut,
            "sow"       : counts,
        }

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            hout[var_name].fill(var_values, weight=event_weights*norm)

            ## Use this for SM samples w/o EFTFitCoefficients
            # hout[var_name].fill(var_values, weight=event_weights[event_selection_mask])

            ## Use this block for EFT samples 
            # hout[var_name].fill(var_values, weight=event_weights_SM*norm)
            # hout[var_name].fill(var_values, weight=event_weights_pt2*norm)
            # hout[var_name].fill(var_values, weight=event_weights_pt3*norm)
            # hout[var_name].fill(var_values, weight=event_weights_pt4*norm)
            # hout[var_name].fill(var_values, weight=event_weights_pt5*norm)
            # hout[var_name].fill(var_values, weight=w2)

        return hout

    def postprocess(self, accumulator):
        return accumulator

