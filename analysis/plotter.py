import pickle #read pickle file
import coffea
from coffea import hist
import topcoffea.modules.HistEFT as HistEFT
import topcoffea.modules.eft_helper as efth
import gzip #read zipped pickle file
import matplotlib.pyplot as plt #plot histograms
from matplotlib.backends.backend_pdf import PdfPages
import topcoffea.modules.utils as utils
import mplhep as hep
import numpy as np

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

#fin = 'TT01j2l_weights_0602.pkl.gz'
#fin = 'TT01j2l_weights.pkl.gz'
#fin = 'weights_test.pkl.gz'
#fin = 'cfg_test.pkl.gz'
#fin = 'TT1j2l_cQj31.pkl.gz'
#fin = "TT2j2l_cQj31.pkl.gz"
#fin = "central_ttbarUL17.pkl.gz"
#fin = 'TT1j2l_RezaReweight.pkl.gz'

flist = ['TT01j2l_S1_weights_0602.pkl.gz', 'TT01j2l_S2_weights_0602.pkl.gz', 'TT01j2l_S3_weights_0602.pkl.gz', 'TT01j2l_S4_weights_0602.pkl.gz', 'TT01j2l_S5_weights_0602.pkl.gz']


###### Define different reweight points ######

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

###### Open pkl file of histograms ######

# with gzip.open(fin) as fin:
#     hin = pickle.load(fin)
#     for k in hin.keys():
#         if k in hists:
#             hists[k]+=hin[k]
#         else:
#             hists[k]=hin[k]


###### Make list of reweight points in the same order as the wc list in the hist ######

def order_rwgt_pts(h,rwgt_dict):
    wc_names = h._wcnames
    rwgt_list = []

    for name in wc_names:
        rwgt_list.append(rwgt_dict[name])

    return rwgt_list

###### Plotting Functions ######

def plot_hist_NOrwgt(hists, name, label):
    h = hists[name]
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=True)
    ax.legend()
    figname = label + '_' + name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

def plot_hist_sm(hists, name, label):
    h = hists[name]
    h.set_sm()
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=True)
    ax.legend()
    figname = label + '_SM_' + name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

def plot_hist_rwgt(hists, name, label, rwgt_dict):
    h = hists[name]
    rwgt = order_rwgt_pts(h, rwgt_dict)
    h.set_wilson_coeff_from_array(rwgt)

    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=True)
    ax.legend()
    figname = label + name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)


for fname in flist:
    if fname.endswith('.pkl.gz'):
        label = fname[:-7]
    else:
        label = fname
    print(label)

    hists = {}

    hists = utils.get_hist_from_pkl(fname, allow_empty=False)
    print(label, hists)

    ###### Plot histograms ######
    for name in hists:
        plot_hist_NOrwgt(hists, name, label)
#    plot_hist_sm(hists, name, label)
#    plot_hist_rwgt(hists, name, label+"_orig_", orig_pts)
#    plot_hist_rwgt(hists, name, label+"_halforig_", halforig_pts)
#    plot_hist_rwgt(hists, name, label+"_qtorig_", qtorig_pts)
#    plot_hist_rwgt(hists, name, label+"_dblorig_", dblorig_pts)

