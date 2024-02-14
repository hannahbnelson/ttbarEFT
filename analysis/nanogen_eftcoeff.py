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

        #print("self._samples", self._samples)
        #print("self._wc_names_lst", self._wc_names_lst)

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

        '''
        print("\n\n GenJet parton flavor: ", events.GenJet.partonFlavour, "\n\n")
        print("\n\n jets_clean fields", jets_clean.partonFlavour, "\n\n")

        bjets = jets_clean[abs(jets_clean.partonFlavour)==5]
        gluons = jets_clean[abs(jets_clean.partonFlavour)==21]
        dquark = jets_clean[abs(jets_clean.partonFlavour)==1]
        uquark = jets_clean[abs(jets_clean.partonFlavour)==2]
        other_jets = jets_clean[(abs(jets_clean.partonFlavour)!=5) & (abs(jets_clean.partonFlavour)!=21) & (abs(jets_clean.partonFlavour)!=1) & (abs(jets_clean.partonFlavour)!=2)]
        print("num of bjets: ", ak.sum(ak.num(bjets)))
        print("num of gluons: ", ak.sum(ak.num(gluons)))
        print("num of d quarks: ", ak.sum(ak.num(dquark)))
        print("num of u quarks: ", ak.sum(ak.num(uquark)))
        print("total number of jets: ", ak.sum(ak.num(jets_clean)))
        print("num of other jets: ", ak.sum(ak.num(other_jets)))
        print("flav or other_jets: ", other_jets.partonFlavour)
        print("\n\n")
        '''

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

        '''
        bjets_cut = bjets[event_selection_mask]
        gluons_cut = gluons[event_selection_mask]
        print("\n\n bjets_cut num: ", ak.sum(ak.num(bjets_cut)))
        print("gluons_cut num: ", ak.sum(ak.num(gluons_cut)))
        print("\n\n")
        '''

        eft_coeffs = eft_coeffs[event_selection_mask]
        coeffs_quad = ak.sum(eft_coeffs, axis = 0)
        print(coeffs_quad)


        return coeffs_quad


    def postprocess(self, accumulator):
        return accumulator
