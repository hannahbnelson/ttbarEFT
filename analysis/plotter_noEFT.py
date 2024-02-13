import pickle #read pickle file
import gzip #read zipped pickle file
import matplotlib.pyplot as plt #plot histograms
import mplhep as hep
import numpy as np
import hist
from hist import Hist

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

# fin = 'central_ttbar.pkl.gz'
fin = 'histos.pkl.gz'

if fin.endswith('.pkl.gz'):
    label = fin[:-7]
else:
    label = fin
print(label)

hists = {}

###### Open pkl file of histograms ######

with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
        if k in hists:
            hists[k]+=hin[k]
        else:
            hists[k]=hin[k]

print(hists)


###### Plotting Functions ######
def plot_newhist(hists, name, label):
    h = hists[name]
    fig, ax = plt.subplots(1,1)
    hep.histplot(h, ax=ax, stack=True, histtype="fill", label=label)
    ax.legend()
    #plt.yscale('log')
    fig.savefig(label + "_" + name + ".png")
    print("Saving histogram to " + label + "_" + name + ".png")
    plt.close(fig)

###### Plot histograms ######
for name in hists:
    plot_newhist(hists, name, "JetFlavTest")

