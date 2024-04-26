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


########################
##### User Inputs  #####
########################

'''
# dim6top ctG test
SM_pts = {"ctG":0.0, "cQq83": 0.0, "cQq81": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctq8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQq13": 0.0, "cQq11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctq1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

ctGp1_pts = {"ctG":1.2, "cQq83": 0.0, "cQq81": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctq8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQq13": 0.0, "cQq11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctq1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

ctGm1_pts = {"ctG":-1.2, "cQq83": 0.0, "cQq81": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctq8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQq13": 0.0, "cQq11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctq1": 0.0, "ctu1": 0.0, "ctd1": 0.0}
# ctG test

SM_pts = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

ctGp1_pts = {"ctGIm": 0.0, "ctGRe":1.2, "cQj38": 0.0, "cQj18": 0.0,
            "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
            "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
            "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

ctGm1_pts = {"ctGIm": 0.0, "ctGRe":-1.2, "cQj38": 0.0, "cQj18": 0.0,
            "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
            "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
            "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

# Reza Starting Points

orig_pts = {"ctGIm": 1.0, "ctGRe":0.7, "cQj38": 9.0, "cQj18": 7.0,
                "cQu8": 9.5, "cQd8": 12.0, "ctj8": 7.0, "ctu8": 9.0,
                "ctd8": 12.4, "cQj31": 3.0, "cQj11": 4.2, "cQu1": 5.5,
                "cQd1": 7.0, "ctj1": 4.4, "ctu1": 5.4, "ctd1": 7.0}

halforig_pts = {"ctGIm": 0.5, "ctGRe":0.35, "cQj38": 4.5, "cQj18": 3.5,
            "cQu8": 4.75, "cQd8": 6.0, "ctj8": 3.5, "ctu8": 4.5,
            "ctd8": 6.2, "cQj31": 1.5, "cQj11": 2.1, "cQu1": 2.75,
            "cQd1": 3.5, "ctj1": 2.2, "ctu1": 2.7, "ctd1": 3.5}

qtorig_pts = {"ctGIm": 0.25, "ctGRe":0.175, "cQj38": 2.25, "cQj18": 1.75,
            "cQu8": 2.375, "cQd8": 3.0, "ctj8": 1.75, "ctu8": 2.25,
            "ctd8": 3.1, "cQj31": 0.75, "cQj11": 1.05, "cQu1": 1.375,
            "cQd1": 1.75, "ctj1": 1.1, "ctu1": 1.35, "ctd1": 1.75}

dblorig_pts = {"ctGIm": 2.0, "ctGRe":1.4, "cQj38": 18.0, "cQj18": 14.0,
                "cQu8": 19.0, "cQd8": 24.0, "ctj8": 14.0, "ctu8": 18.0,
                "ctd8": 24.8, "cQj31": 6.0, "cQj11": 8.4, "cQu1": 11.0,
                "cQd1": 14.0, "ctj1": 8.8, "ctu1": 10.8, "ctd1": 14.0}
'''

SM_pts = {"ctGIm": 0.0, "ctGRe":0.0, "cQj38": 0.0, "cQj18": 0.0,
          "cQu8": 0.0, "cQd8": 0.0, "ctj8": 0.0, "ctu8": 0.0,
          "ctd8": 0.0, "cQj31": 0.0, "cQj11": 0.0, "cQu1": 0.0,
          "cQd1": 0.0, "ctj1": 0.0, "ctu1": 0.0, "ctd1": 0.0}

orig_pts = {"ctGIm": 1.0, "ctGRe":1.0, "cQj38": 3.0, "cQj18": 3.0,
            "cQu8": 3.0, "cQd8": 3.0, "ctj8": 3.0, "ctu8": 3.0,
            "ctd8": 3.0, "cQj31": 3.0, "cQj11": 3.0, "cQu1": 3.0,
            "cQd1": 3.0, "ctj1": 3.0, "ctu1": 3.0, "ctd1": 3.0}

qtorig_pts = {"ctGIm": 0.25, "ctGRe":0.25, "cQj38": 0.75, "cQj18": 0.75,
            "cQu8": 0.75, "cQd8":0.75, "ctj8": 0.75, "ctu8": 0.75,
            "ctd8": 0.75, "cQj31": 0.75, "cQj11": 0.75, "cQu1": 0.75,
            "cQd1": 0.75, "ctj1": 0.75, "ctu1": 0.75, "ctd1": 0.75}

halforig_pts = {"ctGIm": 0.5, "ctGRe":0.5, "cQj38": 1.5, "cQj18": 1.5,
            "cQu8": 1.5, "cQd8": 1.5, "ctj8": 1.5, "ctu8": 1.5,
            "ctd8": 1.5, "cQj31": 1.5, "cQj11": 1.5, "cQu1": 1.5,
            "cQd1": 1.5, "ctj1": 1.5, "ctu1": 1.5, "ctd1": 1.5}

dblorig_pts = {"ctGIm": 2.0, "ctGRe":2.0, "cQj38": 6.0, "cQj18": 6.0,
            "cQu8": 6.0, "cQd8": 6.0, "ctj8": 6.0, "ctu8": 6.0,
            "ctd8": 6.0, "cQj31": 6.0, "cQj11": 6.0, "cQu1": 6.0,
            "cQd1": 6.0, "ctj1": 6.0, "ctu1": 6.0, "ctd1": 6.0}


################################
##### Function Definitions #####
################################

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


###################################
##### Main Analysis Processor #####
###################################

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
            #"weights_orig"          : Hist(hist.axis.Regular(bins = 20, start = 0, stop = 3, name="event weight")),
            #"weights_orig_log"      : Hist(hist.axis.Regular(bins = 65, start = -10, stop = 3, name="log(event weight)")),
            # "weights_SM"            : Hist(hist.axis.Regular(bins = 20, start = 0, stop = 4, name="event weight")),
            # "weights_SM_log"        : Hist(hist.axis.Regular(bins = 65, start = -10, stop = 3, name="log(event weight)")),
            #"weights_halforig"      : Hist(hist.axis.Regular(bins = 50, start = 0, stop = 3, name="event weight")),
            #"weights_halforig_log"  : Hist(hist.axis.Regular(bins = 65, start = -10, stop = 3, name="log(event weight)")),
            #"weights_qtorig"        : Hist(hist.axis.Regular(bins = 20, start = 0, stop = 3, name="event weight")),
            #"weights_qtorig_log"    : Hist(hist.axis.Regular(bins = 65, start = -10, stop = 3, name="log(event weight)")),
            #"weights_dblorig"       : Hist(hist.axis.Regular(bins = 40, start = 0, stop = 8, name="event weight")),
            #"weights_dblorig_log"   : Hist(hist.axis.Regular(bins = 65, start = -10, stop = 3, name="log(event weight)")),
            #"deltaR"                : Hist(hist.axis.Regular(bins=30, start=0, stop=6, name="deltaR")),
            "jet_flav"               : Hist(hist.axis.Regular(bins=23, start=0, stop=23, name="abs(jet flavor)"))
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

        # Extract the EFT quadratic coeffs and optionally use them to calc the coeffs on the w**2 quartic function
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

        jet_flav = abs(jets_clean[event_selection_mask].partonFlavour)

        '''
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


        ######## Delta R ########
        jets_clean = jets_clean[event_selection_mask]
        njets = njets[event_selection_mask]
        dr = []
        for i in range(ak.max(njets)):
            dr_i = jets_clean[njets>=(i+1)][:,i].delta_r(leps_cut[njets>=(i+1)])
            dr.extend(ak.to_list(ak.flatten(dr_i, axis=None)))
        '''


        ######## Histogram of event weights ########
        eft_coeffs_cut = eft_coeffs
        if eft_coeffs is not None:
            eft_coeffs_cut = eft_coeffs[event_selection_mask]

        #wc_lst_SM = order_wc_values(self._wc_names_lst, SM_pts)
        #wc_lst_orig = order_wc_values(self._wc_names_lst, orig_pts)
        #wc_lst_halforig = order_wc_values(self._wc_names_lst, halforig_pts)
        #wc_lst_qtorig = order_wc_values(self._wc_names_lst, qtorig_pts)
        #wc_lst_dblorig = order_wc_values(self._wc_names_lst, dblorig_pts)
       
        #event_weights_SM = calc_event_weights(eft_coeffs_cut, wc_lst_SM)
        #event_weights_orig = calc_event_weights(eft_coeffs_cut, wc_lst_orig)
        #event_weights_halforig = calc_event_weights(eft_coeffs_cut, wc_lst_halforig)
        #event_weights_qtorig = calc_event_weights(eft_coeffs_cut, wc_lst_qtorig)
        #event_weights_dblorig = calc_event_weights(eft_coeffs_cut, wc_lst_dblorig)
        '''

        wc_lst_ctGp1 = order_wc_values(self._wc_names_lst, ctGp1_pts)
        wc_lst_ctGm1 = order_wc_values(self._wc_names_lst, ctGm1_pts)
        
        event_weights_SM_raw = calc_event_weights(eft_coeffs, wc_lst_SM)
        event_weights_ctGp1 = calc_event_weights(eft_coeffs, wc_lst_ctGp1)
        event_weights_ctGm1 = calc_event_weights(eft_coeffs, wc_lst_ctGm1)

        print(" \n \n")
        #print("SM event weights", event_weights_SM)
        print("Sum of SM event weights", np.sum(event_weights_SM_raw))
        print("Sum of ctGp1 event weights", np.sum(event_weights_ctGp1))
        print("Sum of ctGm1 event weights", np.sum(event_weights_ctGm1))
        print("\n \n")

        #print("\n\n")
        #print(np.max(event_weights_dblorig))
        #print("SM events = 0.0: ", sum(event_weights_SM==0.0))
        #print("orig events = 0.0: ", sum(event_weights_orig==0.0))
        #print("half orig events = 0.0: ", sum(event_weights_halforig==0.0))
        #print("qt orig events = 0.0: ", sum(event_weights_qtorig==0.0))
        #print("dbl orig events = 0.0: ", sum(event_weights_dblorig==0.0))
        #print("\n\n")
        '''

        ######## Fill histos ########
        hout = self._histo_dict
        variables_to_fill = {
            #"weights_orig"          : event_weights_orig,
            #"weights_orig_log"      : np.log10(event_weights_orig),
            #"weights_SM"            : event_weights_SM,
            #"weights_SM_log"        : np.log10(event_weights_SM),
            #"weights_halforig"      : event_weights_halforig,
            #"weights_halforig_log"  : np.log10(event_weights_halforig),
            #"weights_qtorig"        : event_weights_qtorig,
            #"weights_qtorig_log"    : np.log10(event_weights_qtorig),
            #"weights_dblorig"       : event_weights_dblorig,
            #"weights_dblorig_log"   : np.log10(event_weights_dblorig),
            #"deltaR"                : dr,
            "jet_flav"               : ak.flatten(jet_flav)
        }

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            hout[var_name].fill(var_values)

        return hout

    def postprocess(self, accumulator):
        return accumulator
