import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

# Main analysis processor
class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        self._histo_dict = {
        	"njets"		: HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "number of jets", 23, 0, 23))
        }

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

        genpart = events.GenPart
        jets = events.GenJet
        njets = ak.num(jets)

        norm = (xsec/sow)
        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])
        event_weights = norm*genw

        hout = self._histo_dict
        variables_to_fill = {
        	"njets"	: njets,
        }

        for var_name, var_values in variables_to_fill.items():
        	fill_info = {
                var_name    : var_values,
                "sample"    : hist_axis_name,
                "weight"    : event_weights,
                "eft_coeff" : eft_coeffs,
            }

        print("fill info: ", fill_info)
        hout[var_name].fill(**fill_info)

        return hout

    def postprocess(self, accumulator):
    	return accumulator

