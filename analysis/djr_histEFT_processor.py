import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')

import hist
from topcoffea.modules.histEFT import HistEFT

from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
# silence warnings due to using NanoGEN instead of full NanoAOD
NanoAODSchema.warn_missing_crossrefs = False


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

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", label=r"Systematic Uncertainty", growth=True)

        # Create the histograms with new scikit hist
        # self._histo_dict = {
        #     "djr_10_all"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("djr_10_all", "DJR  0 to 1", 80, 0, 4)),
        #     "djr_10_0p"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("djr_10_0p", "DJR  0 to 1", 80, 0, 4)),
        #     "djr_10_1p"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("djr_10_1p", "DJR  0 to 1", 80, 0, 4)),
        # }

        self._histo_dict = {
            "djr_01_all"    :HistEFT(
                                proc_axis,
                                hist.axis.Regular(bins=80, start=0, stop=4, name="djr_01_all", label="DJR 0 to 1"),
                                wc_names=wc_names_lst,
                                label="Events"),
            "djr_01_0p"     :HistEFT(
                                proc_axis,
                                hist.axis.Regular(bins=80, start=0, stop=4, name="djr_01_0p", label="DJR 0 to 1"),
                                wc_names=wc_names_lst,
                                label="Events"),
            "djr_01_1p"     :HistEFT(
                                proc_axis,
                                hist.axis.Regular(bins=80, start=0, stop=4, name="djr_01_1p", label="DJR 0 to 1"),
                                wc_names=wc_names_lst,
                                label="Events"),
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

        djr_01 = events.LHEWeight.DJR10
        nMEPartons = events.Generator.nMEPartonsFiltered

        djr_01_0p = djr_01[nMEPartons == 0]
        djr_01_1p = djr_01[nMEPartons == 1]

        wgts = np.ones_like(events['event'])
        if eft_coeffs is None:
            # Basically any central MC samples
            wgts = events["genWeight"]

        ####### Fill Histogram #######
        hout = self._histo_dict

        djr_01_fill_info = {
            "djr_01_all"  	: np.log10(djr_01),
            "process"    	: hist_axis_name,
            "weight"    	: wgts,
            "eft_coeff" 	: eft_coeffs,
        }

        djr_01_0p_fill_info = {
            "djr_01_0p"  	: np.log10(djr_01_0p),
            "process"    	: hist_axis_name,
            "weight"    	: wgts[nMEPartons == 0],
            "eft_coeff" 	: eft_coeffs[nMEPartons == 0],
        }

        djr_01_1p_fill_info = {
            "djr_01_1p"  	: np.log10(djr_01_1p),
            "process"    	: hist_axis_name,
            "weight"    	: wgts[nMEPartons == 1],
            "eft_coeff" 	: eft_coeffs[nMEPartons == 1],
        }

        hout["djr_01_all"].fill(**djr_01_fill_info)
        hout["djr_01_0p"].fill(**djr_01_0p_fill_info)
        hout["djr_01_1p"].fill(**djr_01_1p_fill_info)

        return hout


    def postprocess(self, accumulator):
        return accumulator






