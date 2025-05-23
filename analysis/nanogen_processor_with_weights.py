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

SM_pts = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

# Close to SM but not SM
pt0 = {'ctGIm': 0.1, 'ctGRe':0.1, 'cQj38':0.5, 'cQj18':0.5,
            'cQu8':0.5, 'cQd8':0.5, 'ctj8':0.5, 'ctu8':0.5,
            'ctd8':0.5, 'cQj31':0.5, 'cQj11':0.5, 'cQu1':0.5,
            'cQd1':0.5, 'ctj1':0.5, 'ctu1':0.5, 'ctd1':0.5}

# Close to SM but ctG negative
pt1 = {'ctGIm':- 0.1, 'ctGRe':-0.1, 'cQj38':0.5, 'cQj18':0.5,
            'cQu8':0.5, 'cQd8':0.5, 'ctj8':0.5, 'ctu8':0.5,
            'ctd8':0.5, 'cQj31':0.5, 'cQj11':0.5, 'cQu1':0.5,
            'cQd1':0.5, 'ctj1':0.5, 'ctu1':0.5, 'ctd1':0.5}

# Roughly half of the limits with have, ctG positive
pt2 = {'ctGIm': 0.5, 'ctGRe':0.5, 'cQj38':3.0, 'cQj18':2.0,
            'cQu8':2.0, 'cQd8':4.0, 'ctj8':1.5, 'ctu8':2.2,
            'ctd8':4.0, 'cQj31':1.5, 'cQj11':1.5, 'cQu1':1.5,
            'cQd1':2.5, 'ctj1':1.4, 'ctu1':1.6, 'ctd1':2.5}

# Roughly half of the limits we have, ctG negative
pt3 = {'ctGIm': -0.5, 'ctGRe':-0.5, 'cQj38':3.0, 'cQj18':2.0,
            'cQu8':2.0, 'cQd8':4.0, 'ctj8':1.5, 'ctu8':2.2,
            'ctd8':4.0, 'cQj31':1.5, 'cQj11':1.5, 'cQu1':1.5,
            'cQd1':2.5, 'ctj1':1.4, 'ctu1':1.6, 'ctd1':2.5}

# Close to the limits we have, positive ctG
pt4 = {'ctGIm': 1.0, 'ctGRe':1.0, 'cQj38':6.0, 'cQj18':5.0,
            'cQu8':4.0, 'cQd8':9.0, 'ctj8':3.0, 'ctu8':4.5,
            'ctd8':9.0, 'cQj31':3.0, 'cQj11':3.0, 'cQu1':3.0,
            'cQd1':4.5, 'ctj1':2.5, 'ctu1':3.2, 'ctd1':4.5}

# Larger WC values (roughly double limits we have)
pt5 = {'ctGIm': 3, 'ctGRe':3, 'cQj38':12.0, 'cQj18':10.0,
            'cQu8':8.0, 'cQd8':18.0, 'ctj8':6.0, 'ctu8':9,
            'ctd8':18.0, 'cQj31':6.0, 'cQj11':6.0, 'cQu1':6.0,
            'cQd1':9, 'ctj1':5, 'ctu1':7, 'ctd1':9}

# Really large WC values
pt6 = {'ctGIm': 4, 'ctGRe':4, 'cQj38':12.0, 'cQj18':12.0,
            'cQu8':12.0, 'cQd8':12.0, 'ctj8':12.0, 'ctu8':12,
            'ctd8':12.0, 'cQj31':12.0, 'cQj11':12.0, 'cQu1':12.0,
            'cQd1':12, 'ctj1':12, 'ctu1':12, 'ctd1':12}

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


# Main analysis processor
class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        #print("self._samples", self._samples)
        #print("self._wc_names_lst", self._wc_names_lst)

        # Create the histograms with HistEFT based on coffea.histt
        self._histo_dict = {
            # "tops_pt"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("tops_pt", "pT of the sum of the tops", 50, 0, 1000)),
            # "ht"            : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ht", "hT(Scalar sum of genjet pt)", 50, 0, 1000)),
            # "jets_pt"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("jets_pt", "pT of the sum of the jets", 50, 0, 1000)),
            # "j0pt"          : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("j0pt", "pT of the leading jet", 50, 0, 1000)),
            # "ntops"         : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ntops", "ntops", 10, 0, 10)),
            # "njets"         : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "njets", 10, 0, 10)),
            # "mtt"           : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mtt", "invariant mass of tops", 50, 0, 1000)),
            # "nleps"         : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("nleps", "number of leptons", 10, 0, 10)),
            # "mll"           : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll", "invariant mass of the leptons", 50, 0, 1000)),
            # "weights_SM"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_SM", "event weight", 30, 0, 3)),
            "weights_SM_log"    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_SM_log", "$log_{10}$(event weight)", 70, -6, 1)),
            # "weights_pt0"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt0", "event weight", 30, 0, 3)),
            "weights_pt0_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt0_log", "$log_{10}$(event weight)", 70, -6, 1)),
            # "weights_pt1"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt1", "event weight", 30, 0, 3)),
            "weights_pt1_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt1_log", "$log_{10}$(event weight)", 70, -6, 1)),
            # "weights_pt2"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt2", "event weight", 30, 0, 3)),
            "weights_pt2_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt2_log", "$log_{10}$(event weight)", 70, -6, 1)),
            # "weights_pt3"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt3", "event weight", 30, 0, 3)),
            "weights_pt3_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt3_log", "$log_{10}$(event weight)", 70, -6, 1)),
            # "weights_pt4"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt4", "event weight", 30, 0, 3)),
            "weights_pt4_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt4_log", "$log_{10}$(event weight)", 80, -6, 2)),
            "weights_pt5_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt5_log", "$log_{10}$(event weight)", 80, -6, 2)),
            "weights_pt6_log"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("weights_pt6_log", "$log_{10}$(event weight)", 80, -6, 2)),
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

        # # Standard 2l2j selections
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)

        # selections = PackedSelection()
        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # event_selection_mask = selections.all('2l', '2j')

        # # 2l2j selections & mtt < 700 
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)
        # mtt_selec = ak.fill_none(mtt<700, False)

        # selections = PackedSelection()
        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # selections.add('mtt', mtt_selec)
        # event_selection_mask = selections.all('2l', '2j', 'mtt')

        # # 2l2j selections & 700 <= mtt <= 900
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)
        # mtt_selec1 = ak.fill_none(mtt >=700, False)
        # mtt_selec2 = ak.fill_none(mtt <= 900, False)

        # selections = PackedSelection()
        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # selections.add('mtt1', mtt_selec1)
        # selections.add('mtt2', mtt_selec2)
        # event_selection_mask = selections.all('2l', '2j', 'mtt1', 'mtt2')

        # 2l2j selections mtt > 900
        at_least_two_leps = ak.fill_none(nleps>=2,False)
        at_least_two_jets = ak.fill_none(njets>=2,False)
        mtt_selec = ak.fill_none(mtt > 900, False)

        selections = PackedSelection()
        selections.add('2l', at_least_two_leps)
        selections.add('2j', at_least_two_jets)
        selections.add('mtt', mtt_selec)
        event_selection_mask = selections.all('2l', '2j', 'mtt')


        ######## Variables with Selections ########
        # leps_cut = leps[event_selection_mask]
        # tops_pt_cut = gen_top.sum().pt[event_selection_mask]
        # njets_cut = njets[event_selection_mask]
        # nleps_cut = nleps[event_selection_mask]
        # mtt_cut = mtt[event_selection_mask]
        # ht_cut = ht[event_selection_mask]
        # ntops_cut = ntops[event_selection_mask]
        # jets_pt_cut = jets_clean.sum().pt[event_selection_mask]
        # j0pt_cut = j0.pt[event_selection_mask]
        # mll = (leps_cut[:,0] + leps_cut[:,1]).mass
        
        ######## Normalization ########

        # Normalize by (xsec/sow)
        #lumi = 1000.0*get_lumi(year)
        # norm = (xsec/sow)
        # norm = (1/sow)
        norm = 1.0
        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])

        event_weights = norm*genw

        # ######## Fill histos ########

        hout = self._histo_dict

        eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None

        ######## Fill event weight histos ########
        wc_lst_SM = order_wc_values(self._wc_names_lst, SM_pts)
        wc_lst_pt0 = order_wc_values(self._wc_names_lst, pt0)
        wc_lst_pt1 = order_wc_values(self._wc_names_lst, pt1)
        wc_lst_pt2 = order_wc_values(self._wc_names_lst, pt2)
        wc_lst_pt3 = order_wc_values(self._wc_names_lst, pt3)
        wc_lst_pt4 = order_wc_values(self._wc_names_lst, pt4)
        wc_lst_pt5 = order_wc_values(self._wc_names_lst, pt5)
        wc_lst_pt6 = order_wc_values(self._wc_names_lst, pt6)

        event_weights_SM = calc_event_weights(eft_coeffs_cut, wc_lst_SM)
        event_weights_pt0 = calc_event_weights(eft_coeffs_cut, wc_lst_pt0)
        event_weights_pt1 = calc_event_weights(eft_coeffs_cut, wc_lst_pt1)
        event_weights_pt2 = calc_event_weights(eft_coeffs_cut, wc_lst_pt2)
        event_weights_pt3 = calc_event_weights(eft_coeffs_cut, wc_lst_pt3)
        event_weights_pt4 = calc_event_weights(eft_coeffs_cut, wc_lst_pt4)
        event_weights_pt5 = calc_event_weights(eft_coeffs_cut, wc_lst_pt5)
        event_weights_pt6 = calc_event_weights(eft_coeffs_cut, wc_lst_pt6)

        weights_hist = np.ones_like(event_weights_SM)

        weights_to_fill = {
            # "weights_SM"        : event_weights_SM,
            "weights_SM_log"    : np.log10(event_weights_SM),
            # "weights_pt0"       : event_weights_pt0,
            "weights_pt0_log"   : np.log10(event_weights_pt0),
            # "weights_pt1"       : event_weights_pt1,
            "weights_pt1_log"   : np.log10(event_weights_pt1),
            # "weights_pt2"       : event_weights_pt2,
            "weights_pt2_log"   : np.log10(event_weights_pt2),
            # "weights_pt3"       : event_weights_pt3,
            "weights_pt3_log"   : np.log10(event_weights_pt3),
            # "weights_pt4"       : event_weights_pt4,
            "weights_pt4_log"   : np.log10(event_weights_pt4),
            "weights_pt5_log"   : np.log10(event_weights_pt5),
            "weights_pt6_log"   : np.log10(event_weights_pt6),
        }

        for var_name, var_val in weights_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            fill_info_weights = {
                var_name    : var_val,
                "sample"    : hist_axis_name,
                "weight"    : weights_hist,
                "eft_coeff" : None,
            }

            hout[var_name].fill(**fill_info_weights)

        return hout


    def postprocess(self, accumulator):
        return accumulator

