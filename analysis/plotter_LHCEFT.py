import pickle #read pickle file
import os
import coffea
from coffea import hist
import topcoffea.modules.HistEFT as HistEFT
import topcoffea.modules.eft_helper as efth
import gzip #read zipped pickle file
import matplotlib.pyplot as plt #plot histograms
from matplotlib.backends.backend_pdf import PdfPages
import topcoffea.modules.utils as utils
from topcoffea.scripts.make_html import make_html
import mplhep as hep
import numpy as np

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20,
        }
plt.rcParams.update(params)

orig_pts = {"ctGIm": 1.0, "ctGRe":1.0, "cQj38": 3.0, "cQj18": 3.0,
            "cQu8": 3.0, "cQd8": 3.0, "ctj8": 3.0, "ctu8": 3.0,
            "ctd8": 3.0, "cQj31": 3.0, "cQj11": 3.0, "cQu1": 3.0,
            "cQd1": 3.0, "ctj1": 3.0, "ctu1": 3.0, "ctd1": 3.0}


def plot_hist_sm(hists, name, label, y_label, leg=None):
    h = hists[name]
    h.set_sm()
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=False)
    if leg == None: 
        ax.legend(loc = 'upper right')
    else: 
        ax.legend(leg, loc='upper right')
    # ax.legend(loc = 'upper right', fontsize = 'medium')
    ax.set_ylabel(y_label, fontsize='medium')
    # fig.suptitle("Reweighted to SM")
    figname = label + '_SMrwgt_' + name + '.png'
    fig.savefig(figname)
    print("plot saved to: ", figname)
    plt.close(fig)

def plot_hist_rwgt(hists, name, label, y_label, rwgt_dict=orig_pts, leg=None):
    h = hists[name]
    h.set_wilson_coefficients(**rwgt_dict)
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=False)
    ax.set_ylabel(y_label, fontsize='medium')
    if leg == None: 
        ax.legend(loc = 'upper right')
    else: 
        ax.legend(leg, loc='upper right')
    # fig.suptitle("Reweighted to Starting Point")
    figname = label + '_rwgt2_' + name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

# fin = "LHCEFT_ref_SM.pkl.gz"
fin = "LHCEFT_ref_ref2.pkl.gz"
hists = utils.get_hist_from_pkl(fin, allow_empty=False)

if fin.endswith('.pkl.gz'):
    label = fin[:-7]
else:
    label = fin
print(label)

dr_ylabel = r"$d\sigma \: [pb] \; / \; d\Delta R \: [bin]$"
avg_ylabel = r"$d\sigma \: [pb] \; / \; d p_T \: [GeV]$"
l0pt_ylabel = r"$d\sigma \: [pb] \; / \; d p_T \: [GeV]$"

# legend_label = [r"$t\bar{t}$ (Ref_main)", r"$t\bar{t}$ (SM)", r"$t\bar{t}+$1 jet (Ref_main)", r"$t\bar{t}+$1 jet (SM)"]
legend_label = [r"$t\bar{t}$ (Ref_main)", r"$t\bar{t}$ (Ref_test)", r"$t\bar{t}+$1 jet (Ref_main)", r"$t\bar{t}+$1 jet (Ref_test)"]

# plot_hist_sm(hists, 'dr_leps', label, dr_ylabel, legend_label)
# plot_hist_sm(hists, 'avg_top_pt', label, avg_ylabel, legend_label)
# plot_hist_sm(hists, 'l0pt', label, l0pt_ylabel, legend_label)

plot_hist_rwgt(hists, 'dr_leps', label, dr_ylabel, orig_pts, legend_label)
plot_hist_rwgt(hists, 'avg_top_pt', label, avg_ylabel, orig_pts, legend_label)
plot_hist_rwgt(hists, 'l0pt', label, l0pt_ylabel, orig_pts, legend_label)

