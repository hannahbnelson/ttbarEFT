import pickle #read pickle file
import gzip #read zipped pickle file
import argparse
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

###### Plotting Functions ######

def plot_hist2(hist1, hist2, name, label2):
    h1 = hist1[name]
    h2 = hist2[name]
    fig, ax = plt.subplots()
    hep.histplot(h1, ax=ax, stack=True, label = 'TT01j2l_S1')
    hep.histplot(h2, ax=ax, stack=True, label = 'TT01j2l_S2')
    ax.legend()
    fig.savefig(label2 + "_" + name + ".png")
    plt.close(fig)

def plot_hist1(hists, name, label):
    h = hists[name]
    fig, ax = plt.subplots(1,1)
    hep.histplot(h, ax=ax, stack=True, histtype="fill", label=label)
    ax.legend()
    #plt.yscale('log')
    fig.savefig(label + "_" + name + ".png")
    print("Saving histogram to " + label + "_" + name + ".png")
    plt.close(fig)


def main():
    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--file-path1", default="", help = "The path to the pkl file")
    parser.add_argument("-f2", "--file-path2", default="", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--outname", default="plots", help = "A name for the output file")
    args = parser.parse_args()

    #hist1 = utils.get_hist_from_pkl(args.file_path1,allow_empty=False)
    #hist2 = utils.get_hist_from_pkl(args.file_path2,allow_empty=False)

    #plot_hist2(hist1, hist2, "weights_SM_log", "TT01j2l")

    hist = utils.get_hist_from_pkl("cfg_test.pkl.gz", allow_empty=False)
    print(hist)

if __name__ == "__main__":
    main()
