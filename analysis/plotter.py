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


#flist = ["LHCEFT_ref_SM.pkl.gz"]
#flist = ["LHCEFT_ref_SM_div.pkl.gz"]
flist = ["LHCEFT_ref_SM.pkl.gz"]
#flist = ["LHCEFT_ref_ref2.pkl.gz"]

###### Define different reweight points ######

# WC points for Eddie's samples (wc names slightly changed because of Sergio's ML scripts
# orig_pts = {'ctg':0.7, 'cqq83':9.0, 'cqq81':7.0, 'cqu8':9.5,
#             'cqd8':12.0, 'ctq8':7.0, 'ctu8':9.0, 'ctd8':12.4,
#             'cqq13':4.1, 'cqq11':4.2, 'cqu1':5.5, 'cqd1':7.0,
#             'ctq1':4.4, 'ctu1':5.4, 'ctd1':7.0}

# original dim6top starting point
#orig_pts = {'ctG':0.7, 'cQq83':9.0, 'cQq81':7.0, 'cQu8':9.5,
#            'cQd8':12.0, 'ctq8':7.0, 'ctu8':9.0, 'ctd8':12.4,
#            'cQq13':4.1, 'cQq11':4.2, 'cQu1':5.5, 'cQd1':7.0,
#            'ctq1':4.4, 'ctu1':5.4, 'ctd1':7.0}

# ctq8ref_pts = {'ctG':0, 'cQq83':0, 'cQq81':0, 'cQu8':0,
#                 'cQd8':0, 'ctq8':7.0, 'ctu8':0, 'ctd8':0,
#                 'cQq13':0, 'cQq11':0, 'cQu1':0, 'cQd1':0,
#                 'ctq1':0, 'ctu1':0, 'ctd1':0}

#orig_pts = {"ctGIm": 1.0, "ctGRe":0.7, "cQj38": 9.0, "cQj18": 7.0,
#            "cQu8": 9.5, "cQd8": 12.0, "ctj8": 7.0, "ctu8": 9.0,
#            "ctd8": 12.4, "cQj31": 3.0, "cQj11": 4.2, "cQu1": 5.5,
#            "cQd1": 7.0, "ctj1": 4.4, "ctu1": 5.4, "ctd1": 7.0}

orig_pts = {"ctGIm": 1.0, "ctGRe":1.0, "cQj38": 3.0, "cQj18": 3.0,
            "cQu8": 3.0, "cQd8": 3.0, "ctj8": 3.0, "ctu8": 3.0,
            "ctd8": 3.0, "cQj31": 3.0, "cQj11": 3.0, "cQu1": 3.0,
            "cQd1": 3.0, "ctj1": 3.0, "ctu1": 3.0, "ctd1": 3.0}

# halforig_pts = {"ctGIm": 0.5, "ctGRe":0.35, "cQj38": 4.5, "cQj18": 3.5,
#                 "cQu8": 4.75, "cQd8": 6.0, "ctj8": 3.5, "ctu8": 4.5,
#                 "ctd8": 6.2, "cQj31": 1.5, "cQj11": 2.1, "cQu1": 2.75,
#                 "cQd1": 3.5, "ctj1": 2.2, "ctu1": 2.7, "ctd1": 3.5}

# qtorig_pts = {"ctGIm": 0.25, "ctGRe":0.175, "cQj38": 2.25, "cQj18": 1.75,
#                 "cQu8": 2.375, "cQd8": 3.0, "ctj8": 1.75, "ctu8": 2.25,
#                 "ctd8": 3.1, "cQj31": 0.75, "cQj11": 1.05, "cQu1": 1.375,
#                 "cQd1": 1.75, "ctj1": 1.1, "ctu1": 1.35, "ctd1": 1.75}

# dblorig_pts = {"ctGIm": 2.0, "ctGRe":1.4, "cQj38": 18.0, "cQj18": 14.0,
#                 "cQu8": 19.0, "cQd8": 24.0, "ctj8": 14.0, "ctu8": 18.0,
#                 "ctd8": 24.8, "cQj31": 6.0, "cQj11": 8.4, "cQu1": 11.0,
#                 "cQd1": 14.0, "ctj1": 8.8, "ctu1": 10.8, "ctd1": 14.0}

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
    print(name, h.values())
    hist.plot1d(h, ax=ax, stack=False)
    ax.legend()
    figname = label + '_' + name + '.png'
    fig.suptitle("Reweighted to SM")
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

def plot_hist_sm(hists, name, label):
    h = hists[name]
    h.set_sm()
    # print(name, h.values())
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=False)
    ax.legend(loc = 'upper right', fontsize = 'medium')
    # fig.suptitle("Reweighted to SM")
    figname = label + '_SMrwgt_' + name + '.png'
    fig.savefig(figname)
    print("plot saved to: ", figname)
    plt.close(fig)

def plot_hist_rwgt(hists, name, label, rwgt_dict):
    h = hists[name]
    rwgt = order_rwgt_pts(h, rwgt_dict)
    h.set_wilson_coeff_from_array(rwgt)
    print(name, h.values())
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=False)
    ax.legend()
    fig.suptitle("Reweighted to Starting Point")
    figname = label + name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

def plot_hist_rwgt_dict(hists, name, label, rwgt_dict):
    h = hists[name]
    h.set_wilson_coefficients(**rwgt_dict)
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    hist.plot1d(h, ax=ax, stack=False)
    ax.legend()
    figname = label + '_rwgt2_'+ name + '.png'
    fig.savefig(figname)
    print("Histogram saved to:", figname)
    plt.close(fig)

def plot_hist_weights(hists, name, label, title):
    h = hists[name]
    fig, ax = plt.subplots(1,1) #create an axis for plotting
    # print(name, h.values())
    hist.plot1d(h, ax=ax, stack=False)
    ax.legend()
    figname = label + '.png'
    fig.suptitle(title)
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
    for name in hists: 
        plot_hist_sm(hists, name, label)
        #plot_hist_rwgt_dict(hists, name, label, orig_pts)

    #if "www" in save_dir_path:
    #    make_html(save_dir)




    
