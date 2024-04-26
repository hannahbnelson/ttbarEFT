import coffea
from coffea import hist

import topcoffea.modules.HistEFT as HistEFT
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.utils as utils

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mplhep as hep
import numpy as np


hep.style.use("CMS")
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'legend.fontsize':20}
plt.rcParams.update(params)

def save_fig_as_png(fig, figname):
	fig.savefig(figname+'.png')
	print("figure saved to: ", figname+'.png')

def save_fig_as_pdf(fig, figname):
	fig.savefig(figname+'.pdf')
	print("figure saved to: ", figname+'.pdf')

def plot_single_histo(histo, loc='upper right', title=None):
	# HistEFT based on coffea

	hep.style.use("CMS")
	params = {'axes.labelsize': 25,
	          'axes.titlesize': 25,
	          'legend.fontsize':20}
	plt.rcParams.update(params)

	fig, ax = plt.subplots(1,1)
	hist.plot1d(histo, ax=ax, stack=False, clear=False)
	ax.legend(loc=loc)

	plt.figtext(0.13, 0.89, title, fontsize=25)

	return fig

def make_single_histo(hists, name, label, title=None):
	# HistEFT based on coffea

    histo = hists[name]
    fig = plot_single_histo(histo, title)
    figname = label+'_'+name

    save_fig_as_png(fig, figname)


def make_sm_histo(hists, name, label):
	# HistEFT based on coffea

	histo = hists[name]
	histo.set_sm()
	fig = plot_single_histo(histo, title="Reweighted to SM")
	figname = label+'_SMrwgt_'+name

	save_fig_as_png(fig, figname)


def make_rwgt_histo(hists, name, label, rwgt_dict):
	# HistEFT based on coffea

	histo = hists[name]
	histo.set_wilson_coefficients(**rwgt_dict)
	fig = plot_single_histo(histo, title="Reweighted to Pt1")
	figname = label+'_rwgt_'+name

	save_fig_as_png(fig, figname)


def make_djr01_sm_plot(hists, label, title=None):

	h = hists["djr_10_all"]
	h0 = hists["djr_10_0p"]
	h1 = hists["djr_10_1p"]
	h.set_sm()
	h0.set_sm()
	h1.set_sm()

	fig, ax = plt.subplots(1,1)
	ax.set_yscale('log')

	hist.plot1d(h, stack=False)
	hist.plot1d(h0, stack=False)
	hist.plot1d(h1, stack=False)

	ax.set_xlabel(r"DJR 0 $\rightarrow$ 1")
	ax.legend(["Total", "0 partons", "1 parton"])
	plt.figtext(0.13, 0.89, title, fontsize=25)

	figname = label+'_sm_djr01'
	save_fig_as_png(fig, figname)

def make_djr01_rwgt_plot(hists, rwgt_dict, label, title=None):

	h = hists["djr_10_all"]
	h0 = hists["djr_10_0p"]
	h1 = hists["djr_10_1p"]
	h.set_wilson_coefficients(**rwgt_dict)
	h0.set_wilson_coefficients(**rwgt_dict)
	h1.set_wilson_coefficients(**rwgt_dict)

	fig, ax = plt.subplots(1,1)
	ax.set_yscale('log')

	hist.plot1d(h, stack=False)
	hist.plot1d(h0, stack=False)
	hist.plot1d(h1, stack=False)

	ax.set_xlabel(r"DJR 0 $\rightarrow$ 1")
	ax.legend(["Total", "0 partons", "1 parton"])
	plt.figtext(0.13, 0.89, title, fontsize=25)

	figname = label+'_rwgt_djr01'
	save_fig_as_png(fig, figname)


