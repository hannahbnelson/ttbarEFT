import pickle
import coffea
from coffea import hist

import topcoffea.modules.HistEFT as HistEFT
import topcoffea.modules.utils as utils
import ttbarEFT.modules.plotting_tools as pt

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

# #### Make Kinematic Plots ####
# fin = "central_TT01j2l_S4v8.pkl.gz"
# if fin.endswith('.pkl.gz'):
#     label = fin[:-7]
# else:
#     label = fin
# hists = utils.get_hist_from_pkl(fin, allow_empty=False)

# for name in hists:
# 	pt.save_sm_histo(hists, name, label)


#### Make DJR Plots ####
fin = "TT01j2l_S4v8_djr.pkl.gz"
if fin.endswith('.pkl.gz'):
    label = fin[:-7]
else:
    label = fin
hists = utils.get_hist_from_pkl(fin, allow_empty=False)

pt.make_djr01_plot(hists, label)


# #### Make Event Weight Plots ####
# fin = "TT01j2l_S4v8_weights.pkl.gz"
# if fin.endswith('.pkl.gz'):
#     label = fin[:-7]
# else:
#     label = fin

# for name in hists:
# 	pt.make_single_histo(hists, name, title = name, loc='upper left')



