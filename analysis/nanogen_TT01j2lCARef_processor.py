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

        print("\n\n")
        print("self._samples", self._samples)
        print("self._wc_names_lst", self._wc_names_lst)
        print("\n\n")

        # Create the histograms with new scikit hist
        self._histo_dict = {
            "tops_pt"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("tops_pt", "$p_T$ of the sum of the tops", 70, 0, 700)),
            "avg_top_pt"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("avg_top_pt", "average top $p_T$ [GeV]", 40, 0, 400)),
            "l0pt"         : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("l0pt", "leading lepton $p_T$ [GeV]", 40, 0, 400)),
            "dr_leps"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("dr_leps", "$\Delta R$ (leading lepton, subleading lepton)", 40, 0, 8)),
            "ht"           : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ht", "hT(Scalar sum of genjet pt)", 50, 0, 1000)),
            "jets_pt"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("jets_pt", "pT of the sum of the jets", 50, 0, 1000)),
            "j0pt"         : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("j0pt", "pT of the leading jet", 50, 0, 1000)),
            "ntops"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ntops", "ntops", 10, 0, 10)),
            "njets"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "njets", 10, 0, 10)),
            "mtt"          : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mtt", "invariant mass of tops", 65, 0, 1300)),
            "nleps"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("nleps", "number of leptons", 10, 0, 10)),
            "mll"          : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll", "invariant mass of the leptons", 40, 0, 800)),
            "LHE_HT"       : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("LHE_HT", "LHE_HT", 50, 0, 1000)),
            "LHE_HTIncoming": HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("LHE_HTIncoming", "LHE_HTIncoming", 25, 0, 1000)),
            "sow"           : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("sow", "sum of weights", 1, 0, 2)),
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
        sow_after_selec = self._samples[dataset]['nSumOfWeightsAfterSelec']

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

        # Standard 2l2j selections
        at_least_two_leps = ak.fill_none(nleps>=2,False)
        at_least_two_jets = ak.fill_none(njets>=2,False)

        selections = PackedSelection()
        selections.add('2l', at_least_two_leps)
        selections.add('2j', at_least_two_jets)
        event_selection_mask = selections.all('2l', '2j')
        # mtt_norm = sow_after_selec

        # # 2l2j selections & mtt < 700 
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)
        # mtt_selec = ak.fill_none(mtt<700, False)

        # selections = PackedSelection()
        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # selections.add('mtt', mtt_selec)
        # event_selection_mask = selections.all('2l', '2j', 'mtt')
        # mtt_norm = self._samples[dataset]['nSumOfWeights_mtt_0_700']

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
        # mtt_norm = self._samples[dataset]['nSumOfWeights_mtt_700_900']

        # # 2l2j selections mtt > 900
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)
        # mtt_selec = ak.fill_none(mtt > 900, False)

        # selections = PackedSelection()
        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # selections.add('mtt', mtt_selec)
        # event_selection_mask = selections.all('2l', '2j', 'mtt')
        # mtt_norm = self._samples[dataset]['nSumOfWeights_mtt_900_inf']


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

        ######## Normalization ########

        # Normalize by (xsec/sow)
        #lumi = 1000.0*get_lumi(year)
        norm = (xsec/sow)
        # norm = (1/sow_after_selec)
        # norm = (1/sow)
        # norm = (1/200)
        # norm = 1

        # norm = 1/mtt_norm

        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])

        event_weights = norm*genw

        counts = np.ones_like(events['event'])[event_selection_mask]

        ######## Fill histos ########

        hout = self._histo_dict

        variables_to_fill = {
            "tops_pt"   : tops_pt_cut,
            "l0pt"      : ak.flatten(l0pt_cut),
            "dr_leps"   : dr_cut,
            "avg_top_pt": avg_top_pt_cut,
            "njets"     : njets_cut,
            "nleps"     : nleps_cut,
            "mtt"       : mtt_cut,
            "ht"        : ht_cut,
            "ntops"     : ntops_cut,
            "jets_pt"   : jets_pt_cut,
            "j0pt"      : ak.flatten(j0pt_cut),
            "mll"       : mll,
            "LHE_HT"    : lhe_ht,
            "LHE_HTIncoming" : lhe_htincoming, 
            "sow"       : counts,
        }

        eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None
        eft_w2_coeffs_cut = eft_w2_coeffs[event_selection_mask] if eft_w2_coeffs is not None else None

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            fill_info = {
                var_name    : var_values,
                "sample"    : hist_axis_name,
                "weight"    : event_weights[event_selection_mask],
                "eft_coeff" : eft_coeffs_cut,
                "eft_err_coeff" : eft_w2_coeffs_cut,
            }

            hout[var_name].fill(**fill_info)

        return hout


    def postprocess(self, accumulator):
        return accumulator
