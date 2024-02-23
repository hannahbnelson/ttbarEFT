import os
import argparse
# import pickle #read pickle file
import coffea
from coffea import hist
import topcoffea.modules.HistEFT as HistEFT
import topcoffea.modules.eft_helper as efth
# import gzip #read zipped pickle file
import matplotlib.pyplot as plt #plot histograms
from matplotlib.backends.backend_pdf import PdfPages
import topcoffea.modules.utils as utils
from topcoffea.scripts.make_html import make_html
import mplhep as hep
import numpy as np

hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

def get_hist(fname, hist_name):
    hists = utils.get_hist_from_pkl(fname, allow_empty=False)
    h = hists[hist_name]
    return h

def calc_weight_array(hist, wc_range, wc_name):
    '''
    Calculate the sum of event weights for a range of values of a 
    single wc while all others are frozen at 0. 
    Parameters
    ----------
    hist : single HistEFT histogram
        needs to be a 1 bin histogram in order to get the correct output
    wc_range : list
        List of wc values to calculate histogram event weight at
        eg. np.arange(min, max, increment)
    wc_lst : list
        list of all WC contained in the histogram
    wc_name : str
        single wc that will be scanned

    Returns: 
        A nested list of values
        [[wc_range], [hist.values() at each point in wc_range]]
    '''

    wc_lst = hist._wcnames

    weight_pts = [] 
    weights = []
    wc_vals = {}
    
    # set all initial WC values to 0.0
    for item in wc_lst:
        wc_vals[item] = 0.0

    # loop through different values of the wc coeff, get sum of event weights
    for i in wc_range:
        wc_vals.update({wc_name:i})
        hist.set_wilson_coefficients(**wc_vals)
        weight = hist.values()
        weights.extend([list(w)[0] for w in weight.values()])

    weight_pts.append(wc_range)
    weight_pts.append(weights)
        
    return weight_pts

def make_1d_quad_plot(files, hist_name, wc_range, wc_name):
    '''
    Make 1d quadratic plot of wc value versus total event weight
    Parameters
    ----------
    files : list
        List of .pkl.gz files to run over
    hist_name : str
        Histogram name to use to get event weight values
    wc_range : list
        List of wc values to calculate histogram event weight at
    wc_name : str
        single wc that will be scanned
    '''
    
    plot_vals = {}

    for fname in files: 
        if fname.endswith('.pkl.gz'):
            label = fname[:-7]
        else: 
            label = fname
        hist = get_hist(fname, hist_name)
        weights = calc_weight_array(hist, wc_range, wc_name)
        plot_vals[label] = weights
    
    fig, ax = plt.subplots()
    for item in plot_vals:
        ax.plot(plot_vals[item][0], plot_vals[item][1], label = item)

    ax.legend()
    fig.suptitle(wc_name)
    figname = "quad_1d_"+wc_name+".png"
    fig.savefig(figname)
    print("plot saved to: ", figname)
    plt.close(fig)


def make_1d_quad_plot_with_scatter(files, save_dir, hist_name, wc_range, wc_name, scatter_lst):
    '''
    Make 1d quadratic plot of wc value versus total event weight
    Parameters
    ----------
    files : list
        List of .pkl.gz files to run over
    hist_name : str
        Histogram name to use to get event weight values
    wc_range : list
        List of wc values to calculate histogram event weight at
    wc_name : str
        single wc that will be scanned
    manual_lst : list 
        [[x-values],[y-values]]
    '''
    
    plot_vals = {}

    for fname in files: 
        if fname.endswith('.pkl.gz'):
            label = fname[:-7]
        else: 
            label = fname
        hist = get_hist(fname, hist_name)
        weights = calc_weight_array(hist, wc_range, wc_name)
        plot_vals[label] = weights
    
    fig, ax = plt.subplots()
    for item in plot_vals:
        ax.plot(plot_vals[item][0], plot_vals[item][1], label = item)

    ax.scatter(scatter_lst[0], scatter_lst[1], label = "Dedicated")
    
    ax.legend()
    # ax.legend(loc=(1.04, 0.5))
    ax.set_xlabel(wc_name)
    ax.set_ylabel(r"$\sigma_{NP} /\ \sigma_{SM}$")
    plt.grid(True)
    figname = "quad_1d_"+wc_name+".png"
    fig.savefig(os.path.join(save_dir,figname))
    print("plot saved to: ", figname)
    plt.close(fig)


def get_points_from_txt(fname):
    
    with open(fname, "r") as f: 
        lines = f.read().splitlines() 
    f.close() 

    dedicated = {}
    i = 0

    while i < len(lines):
        first = lines[i].split("=")
        second = lines[i+1].split("+-")

        name = first[0]
        xval = float([first[1]][0])
        yval = float([second[0]][0])

        if name not in dedicated.keys():
            dedicated[name] = [[xval], [yval]]
        else: 
            dedicated[name][0].append(xval)
            dedicated[name][1].append(yval)
        i += 2

    return dedicated


# TODO: add a check that all files have the same set of wc names, 
# otherwise not all of the plots will be able to be made

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--files', '-f', action='extend', nargs='+', required = True, help = "Specify a list of pkl.gz to run over.")
    parser.add_argument('--hist-name', default = 'sow', help = 'Which histogram to use')
    parser.add_argument('--wc-range', default = 1.0, type = float, help = 'Range for wc calculated. Plot created for [-num, num).')
    parser.add_argument('--wc-name', action='extend', nargs='+', default = None, help = 'WC names to make plots for')
    parser.add_argument('--outpath',  default=".", help = "The path the output files should be saved to")

    args = parser.parse_args()

    # Set variables from command line arguments
    files = args.files 
    hist_name = args.hist_name
    wc_max = args.wc_range
    wc_name = args.wc_name
    save_dir_path = args.outpath

    # Get full list of possible wc names from the first file
    temp_hist = get_hist(files[0], hist_name)
    temp_wc_lst = temp_hist._wcnames

    # Fill wc_name list. The default is to make plots for all wc names in the hist, 
    # but can be overridden with a list from command line input 
    if wc_name == None:
        wc_list = temp_wc_lst
    else:
        wc_list = wc_name

    fname = "MgXS.txt"
    scatter_dict = get_points_from_txt(fname)

    for wc in wc_list:
        if wc == "ctGRe" or "ctGIm":
            wc_range = np.arange(-1.0, 1.0, 0.25)
       
        else:
            np.arange(-wc_max, wc_max, 0.5)

        scatter_xvals = scatter_dict[wc][0]
        scatter_yvals = np.divide(np.array(scatter_dict[wc][1]), 49.41)
        scatter_lst = [scatter_xvals, scatter_yvals]
        make_1d_quad_plot_with_scatter(files, save_dir_path, hist_name, wc_range, wc, scatter_lst)

    # Make an index.html file if saving to web area
    if "www" in save_dir_path:
        make_html(save_dir_path)

    # Loop through all wc in list and make a 1d quadratic plot for each
    # for wc in wc_list: 
    #     make_1d_quad_plot(files, hist_name, wc_range, wc)





