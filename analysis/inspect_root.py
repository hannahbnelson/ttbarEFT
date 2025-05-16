import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from topcoffea.modules import utils
import topcoffea.modules.eft_helper as efth

#fname = '/project01/ndcms/hnelson2/ttbarEFT/nanoGen/LHCEFT/TT01j2lCARef/nanoGen_660.root'
fname = '/cms/cephfs/data/store/user/hnelson2/mc/central_mini2nano_test/TTto2L2Nu_1Jets_smeft_MTT_700to900/nanoAOD_TTto2L2Nu_1Jets_smeft_MTT_700to900/NAOD-00000_4.root'

# Load in events from root file
events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v6,
    metadata={"dataset": "TT01j2l"},
).events()

wc_lst = utils.get_list_of_wc_names(fname)
# print(wc_lst)

print('All branches: ', events.fields, '\n')

print('LHEReweightingWeight: ', events.LHEReweightingWeight.fields)
print('LHEPdfWeight: ', events.LHEPdfWeight.fields)
print('LHE: ', events.LHE.fields)
print('LHEScaleWeight: ', events.LHEScaleWeight.fields)
print('LHEPart: ', events.LHEPart.fields)
print('LHEWeight: ', events.LHEWeight.fields)

# Inside LHE Part, there are no top quarks (count_yes is 0 after looking at events)
count_no = 0
count_yes = 0
for i in range(100):
    print(events.LHEPart[i].pdgId)
    for j in range(8):
        if abs(events.LHEPart[i][j].pdgId) == 6:
            count_yes += 1
        else: count_no += 1

print("no: ", count_no)
print("yes: ", count_yes)
			
