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
    # for item in wc_lst:
    #     wc_vals[item] = 0.0

    # S0 starting point 
    wc_vals= {"ctGIm": -0.5, "ctGRe":-0.5, "cQj38":1.5, "cQj18":1.5, 
            "cQu8":1.5, "cQd8":1.5, "ctj8":1.5, "ctu8":1.5, 
            "ctd8":1.5, "cQj31":1.5, "cQj11":1.5, "cQu1":1.5, 
            "cQd1":1.5, "ctj1":1.5, "ctu1":1.5, "ctd1":1.5}

    # S1 starting point
    # wc_vals= {"ctGIm": 0.7, "ctGRe":0.7, "cQj38":9.0, "cQj18":7.0, 
    #         "cQu8":9.5, "cQd8":12.0, "ctj8":7.0, "ctu8":9.0, 
    #         "ctd8":12.4, "cQj31":3.0, "cQj11":4.2, "cQu1":5.5, 
    #         "cQd1":7.0, "ctj1":4.4, "ctu1":5.4, "ctd1":7.0}

    # S2 starting point 
    # wc_vals= {"ctGIm": 1.0, "ctGRe":1.0, "cQj38":3.0, "cQj18":3.0, 
    #         "cQu8":3.0, "cQd8":3.0, "ctj8":3.0, "ctu8":3.0, 
    #         "ctd8":3.0, "cQj31":3.0, "cQj11":3.0, "cQu1":3.0, 
    #         "cQd1":3.0, "ctj1":3.0, "ctu1":3.0, "ctd1":3.0}


    # loop through different values of the wc coeff, get sum of event weights
    for i in wc_range:
        wc_vals.update({wc_name:i})
        hist.set_wilson_coefficients(**wc_vals)
        weight = hist.values()
        weights.extend([list(w)[0] for w in weight.values()])

    weight_pts.append(wc_range)
    weight_pts.append(weights)
        
    return weight_pts

def make_1d_quad_plot(files, save_dir, hist_name, wc_max, wc_name):
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

    wc_range = np.arange(-wc_max, wc_max+0.5, 0.5)

    for fname in files: 
        if fname.endswith('.pkl.gz'):
            label = fname[:-7]
        else: 
            label = fname
        hist = get_hist(fname, hist_name)
        weights = calc_weight_array(hist, wc_range, wc_name)
        plot_vals[label] = weights
    
    fig, ax = plt.subplots()

    max_val = 0

    for item in plot_vals:
        if item == "LHCEFT_TT01j2l_ref_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(S1)")
        elif item == "LHCEFT_TT01j2l_rob_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(S2)")
        elif item == "TT01j2lCARef_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l")
        elif item == "TT01j2lCARef_sow_norm": 
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2lCARef")
        elif item == "TT01j2lCARef_sow_0_700":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(mtt<700)")
        elif item == "TT01j2lCARef_sow_700_900":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(700<mtt<900)")
        elif item == "TT01j2lCARef_sow_900_inf":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(mtt>900)")
        else:
            ax.plot(plot_vals[item][0], plot_vals[item][1], label = item)
        new_val = max(plot_vals[item][1])
        if new_val >= max_val:
            max_val = new_val

    ax.legend(loc = 'upper right', fontsize='large')
    ax.set_xlim([-wc_max, wc_max])
    # ax.set_ylim([0.8, max_val+0.5])
    ax.set_xlabel(wc_name, fontsize = 'large')
    ax.set_ylabel(r"$\sigma_{SMEFT} /\ \sigma_{SM}$", fontsize = 'large')
    plt.figtext(0.14, 0.89, r"$pp \rightarrow t\bar{t} \rightarrow l^+ \nu_l b \;\; l^- \bar{\nu_l} \bar{b}$", fontsize='large')
    plt.figtext(0.72, 0.89,"(13 TeV)", fontsize = 'large')

    figname = "S0stpt_quad_1d_mtt_"+wc_name+".png"
    # figname = "S1stpt_quad_1d_mtt_"+wc_name+".png"
    # figname = "S2stpt_quad_1d_mtt_"+wc_name+".png"
    fig.savefig(os.path.join(save_dir,figname))
    print("plot saved to: ", os.path.join(save_dir,figname))
    plt.close(fig)


def make_1d_quad_plot_with_scatter(files, save_dir, hist_name, wc_max, wc_name, scatter_lst):
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
        [[x-values],[y-values],[uncertainty on y-values]]
    '''

    wc_range = np.arange(-wc_max, wc_max+0.5, 0.5)
    
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

    ax.scatter(scatter_lst[0], scatter_lst[1], label = "Dedicated Point")
    ax.errorbar(scatter_lst[0], scatter_lst[1], yerr = scatter_lst[2], xerr = None, capsize=5, ls='none')

    for item in plot_vals:
        if item == "LHCEFT_TT0j2l_Ref_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="LO (sample 1)")
        elif item == "LHCEFT_TT0j2l_Ref2_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1],label="LO (sample 2)")
        elif item == "LHCEFT_TT01j2l_ref_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(S1)")
        elif item == "LHCEFT_TT01j2l_rob_sow":
            ax.plot(plot_vals[item][0], plot_vals[item][1], label="TT01j2l(S2)")
        else:
            ax.plot(plot_vals[item][0], plot_vals[item][1], label = item)
    
    ax.legend(loc = 'upper right', fontsize='medium')
    plt.figtext(0.14, 0.89, r"$pp \rightarrow t\bar{t} \rightarrow l^+ \nu_l b \;\; l^- \bar{\nu_l} \bar{b}$", fontsize='medium')
    plt.figtext(0.72, 0.89,"(13 TeV)", fontsize = 'medium')
    # ax.legend(loc=(1.04, 0.5))
    ax.set_xlim([-wc_max, wc_max])
    ax.set_xlabel(wc_name, fontsize = 'large')
    ax.set_ylabel(r"$\sigma_{SMEFT} /\ \sigma_{SM}$", fontsize = 'large')
    figname = "quad_1d_"+wc_name+".pdf"
    fig.savefig(os.path.join(save_dir,figname))
    print("plot saved to: ", os.path.join(save_dir,figname))
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
        ysigma = float(second[1][:-3])

        if name not in dedicated.keys():
            dedicated[name] = [[xval], [yval], [ysigma]]
        else: 
            dedicated[name][0].append(xval)
            dedicated[name][1].append(yval)
            dedicated[name][2].append(ysigma)   
        i += 2

    return dedicated


# TODO: add a check that all files have the same set of wc names, 
# otherwise not all of the plots will be able to be made

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--files', '-f', action='extend', nargs='+', required = True, help = "Specify a list of pkl.gz to run over.")
    parser.add_argument('--hist-name', default = 'sow_norm', help = 'Which histogram to use')
    parser.add_argument('--wc-range', default = 6.0, type = float, help = 'Range for wc calculated. Plot created for [-num, num).')
    parser.add_argument('--wc-name', action='extend', nargs='+', default = None, help = 'WC names to make plots for')
    parser.add_argument('--outpath',  default=".", help = "The path the output files should be saved to")
    parser.add_argument('--html', action='store_true', help = "Make an html page for the save dir")
    parser.add_argument('--scatter', action='store_true', help="Make quad plots with scatter plot overlay that is made with points from txt file")

    args = parser.parse_args()

    # Set variables from command line arguments
    files = args.files 
    hist_name = args.hist_name
    wc_max = args.wc_range
    wc_name = args.wc_name
    save_dir_path = args.outpath
    html_page = args.html
    scatter = args.scatter

    print("Making plots using the pkl files: \n")
    for f in files: 
        print(f, "\n")

    # Get full list of possible wc names from the first file
    temp_hist = get_hist(files[0], hist_name)
    temp_wc_lst = temp_hist._wcnames

    # Fill wc_name list. The default is to make plots for all wc names in the hist, 
    # but can be overridden with a list from command line input 
    if wc_name == None:
        wc_list = temp_wc_lst
    else:
        wc_list = wc_name

    if scatter:
        fname = "MgXS.txt"
        scatter_dict = get_points_from_txt(fname)

        for wc in wc_list:
            scatter_xvals = scatter_dict[wc][0]
            scatter_yvals = np.divide(np.array(scatter_dict[wc][1]), 49.41)
            scatter_sigma = np.array(scatter_dict[wc][2])
            const = 49.41
            sigma_const = 0.3654
            sigma_y= np.multiply(scatter_yvals, (np.sqrt(np.add(np.square(np.divide(scatter_sigma, scatter_dict[wc][1])),np.square(np.divide(sigma_const, const))))))

            scatter_lst = [scatter_xvals, scatter_yvals, sigma_y]
            if wc == 'ctGRe':
                make_1d_quad_plot_with_scatter(files, save_dir_path, hist_name, 1.0, wc, scatter_lst)
            else: 
                make_1d_quad_plot_with_scatter(files, save_dir_path, hist_name, wc_max, wc, scatter_lst)
    else:
        for wc in wc_list:
            if wc == 'ctGRe':
                make_1d_quad_plot(files, save_dir_path, hist_name, 2.0, wc)
            else:
                make_1d_quad_plot(files, save_dir_path, hist_name, wc_max, wc)


    # Make an index.html file if saving to web area
    if html_page:
        if "www" in save_dir_path:
            make_html(save_dir_path)

