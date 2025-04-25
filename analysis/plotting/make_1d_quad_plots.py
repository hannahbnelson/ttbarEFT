import os
import argparse
import numpy as np

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.utils as utils

import mplhep as hep
import matplotlib.pyplot as plt

import plotting_tools_histEFT as plotTools 


def make_scatter_list_oneWC(scatter_dict, norm, norm_uncert):
	'''
	scatter_dict must be a single entry dictionary, with only the information for a single WC
	'''
	scatter_xvals = scatter_dict[0]
	scatter_yvals = np.divide(np.array(scatter_dict[1]), norm)
	scatter_sigma = np.array(scatter_dict[2])
	sigma_y = np.multiply(scatter_yvals, (np.sqrt(np.add(np.square(np.divide(scatter_sigma, scatter_dict[1])),np.square(np.divide(norm_uncert, norm))))))

	return [scatter_xvals, scatter_yvals, sigma_y]


def make_samples_dict(files, wc_name, sample_names=None, hist_name='sow_norm', wc_max=6.0):
	
    samples = {}

    for fname in files: 
        if sample_names is not None:
            label = sample_names[fname]
        elif fname.endswith('.pkl.gz'):
            label = fname[:-7]
        else: 
            label = fname
            
        wc_range = np.arange(-wc_max, wc_max+0.5, 0.1)
        h = plotTools.get_single_hist(fname, hist_name)
        norm = h.as_hist({}).values()[0] #get SM xsec of the sample and use this for normalization
        weights = plotTools.calc_sow_array(h, wc_range, wc_name)

        if norm != 1.0:
            weights[1] = np.divide(weights[1], norm)

        samples[label] = weights

    return samples

def make_scatter_plot(samples, scatter_lst, wc_max, wc, proc_text='', energy='(13TeV)', outpath='.', scan_type='frozen'):

    fig, ax = plotTools.make_1d_quad_plot_with_scatter(samples, scatter_lst, wc_max, wc)
    plt.figtext(0.72, 0.89, energy, fontsize = 'medium')
    plt.figtext(0.14, 0.89, proc_text, fontsize='medium')

    figname = f"quad1d_scatter_{scan_type}_{wc}.pdf"
    fig.savefig(os.path.join(outpath,figname))
    print(f"plot saved to {os.path.join(outpath,figname)}")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--files', '-f', action='extend', nargs='+', required = True, help = "Specify a list of pkl.gz to run over.")
    parser.add_argument('--hist-name', default = 'sow_norm', help = 'Which histogram to use')
    parser.add_argument('--wc-range', default = 6.0, type = float, help = 'Range for wc calculated. Plot created for [-num, num).')
    parser.add_argument('--wc-names', action='extend', nargs='+', default = None, help = 'WC names to make plots for')
    parser.add_argument('--outpath',  default=".", help = "The path the output files should be saved to")
    parser.add_argument('--html', action='store_true', help = "Make an html page for the save dir")
    parser.add_argument('--scatter', action='store_true', help="Make quad plots with scatter plot overlay that is made with points from txt file")
    parser.add_argument('--scatter-file', default='MgXS.txt', help='')
    parser.add_argument('--scatter-norm', default=1.0, help="Value to normalize scatter values to")
    parser.add_argument('--norm-sigma', default=1.0, help="uncertainty on normalized value")
    parser.add_argument('--ttbar', action='store_true', help="Make plots for ttbar")
    parser.add_argument('--tW', action='store_true', help="Make plots for tW")
    parser.add_argument('--frozen', action='store_true', help="Make frozen plots")
    parser.add_argument('--profiled', action='store_true', help="Make profiled plots")

    args = parser.parse_args()

    # Set variables from command line arguments
    files = args.files 
    hist_name = args.hist_name
    wc_max = args.wc_range
    wc_name = args.wc_names
    save_dir_path = args.outpath
    html_page = args.html
    scatter = args.scatter
    scatter_file = args.scatter_file
    const = args.scatter_norm
    sigma_const = args.norm_sigma
    outpath = args.outpath

    # get list of wcs to loop through (one plot per wc)
    if wc_name is None: 
    	temp_hist = plotTools.get_single_hist(files[0], hist_name)
    	wc_list = list(temp_hist._wc_names.keys())
    else: 
    	wc_list = wc_names

    tW_proc_text = r"$pp \rightarrow t l^{-} \bar{\nu_l} \rightarrow l^+ \nu_l b \;\; l^- \bar{\nu_l} \bar{b}$"
    ttbar_proc_text = r"$pp \rightarrow t\bar{t} \rightarrow l^+ \nu_l b \;\; l^- \bar{\nu_l} \bar{b}$"

    # if st_pt is not None:
    #     scan_type = 'profiled'
    # else: scan_type = 'frozen'
    scan_type = 'frozen'

    scatter_dict = plotTools.read_MGstandalone_txt(scatter_file)
    for wc in wc_list:   

        if wc == 'ctGRe' or wc=='ctGIm': 
            wc_lim = 1.0
        else: 
            wc_lim = wc_max

        # 5.63 += 0.01903 is the standAloneMG value for the SM xsec
        scatter_lst = make_scatter_list_oneWC(scatter_dict[wc], 5.63, 0.01903)
        samples = make_samples_dict(files, wc_name=wc, hist_name=hist_name, wc_max=wc_lim)
        make_scatter_plot(samples, scatter_lst, wc_lim, wc, proc_text=tW_proc_text, scan_type=scan_type, outpath=outpath)

