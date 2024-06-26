import pickle #read pickle file
import gzip #read zipped pickle file
import matplotlib.pyplot as plt #plot histograms
import mplhep as hep
import numpy as np
import hist
from hist import Hist
import topcoffea.modules.utils as utils

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

# fin = 'central_ttbar.pkl.gz'
#fin = 'sow_S1.pkl.gz'
#fin = 'allSamplesJetFlav.pkl.gz'
#fin = 'centralJetFlav.pkl.gz'
fin = 'SMweights_test.pkl.gz'

if fin.endswith('.pkl.gz'):
    label = fin[:-7]
else:
    label = fin
print(label)

hists = {}

###### Open pkl file of histograms ######

hists = utils.get_hist_from_pkl(fin, allow_empty=False)

# with gzip.open(fin) as fin:
#     hin = pickle.load(fin)
#     for k in hin.keys():
#         if k in hists:
#             hists[k]+=hin[k]
#         else:
#             hists[k]=hin[k]

# print(hists)


###### Plotting Functions ######
def plot_newhist(hists, name, label):
    h = hists[name]
    fig, ax = plt.subplots(1,1)
    hep.histplot(h, ax=ax, stack=False, histtype="fill", label=label)
    ax.legend()
    fig.suptitle("Reweighted to the Standard Model")
    #plt.yscale('log')
    fig.savefig(label + "_" + name + ".png")
    print("Saving histogram to " + label + "_" + name + ".png")
    plt.close(fig)

###### Plot histograms ######
for name in hists:
    plot_newhist(hists, name, "label")

