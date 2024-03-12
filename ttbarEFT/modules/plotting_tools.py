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
	print("figure saved to: ", figname)

def save_fig_as_pdf(fig, figname):
	fig.savefig(figname+'.pdf')
	print("figure saved to: ", figname)

def make_single_histo(histo, loc='upper right', fontsize='medium', title=None):
	
	fig, ax = plt.subplots(1,1)
	hist.plot1d(histo, ax=ax, stack=False, clear=False)
	ax.legend(loc=loc, fontsize=fontsize)

	if title is not None:
		fig.suptitle(title, fontsize=fontsize)

	return fig

def save_single_histo(hists, name, label, loc='upper right', fontsize='medium', title=None):
	# HistEFT based on coffea

	histo = hists[name]

	fig = make_single_histo(histo, loc, fontsize, title)
	figname = label+'_'+name+'.png'
	fig.savefig(figname)
	print("figure saved to: ", figname)
	plt.close(fig)


def save_sm_histo(hists, name, label, loc='upper right', fontsize='medium'):
	# HistEFT based on coffea

	histo = hists[name]
	histo.set_sm()

	fig = make_single_histo(histo, loc, fontsize)
	figname = label+'_SMrwgt_'+name+'.png'
	fig.savefig(figname)
	print("figure saved to: ", figname)
	plt.close(fig)


def save_rwgt_histo(hists, name, label, rwgt_dict, loc='upper right', fontsize='medium'):
	# HistEFT based on coffea

	histo = hists[name]
	histo.set_wilson_coefficients(**rwgt_dict)

	fig = make_single_histo(histo, loc, fontsize)
	figname = label+'_rwgt_'+name+'.png'
	fig.savefig(figname)
	print("figure saved to: ", figname)
	plt.close(fig)

def make_djr01_plot(hists, label):

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

	figname = label+'_djr01.png'
	fig.savefig(figname)
	print("figure saved to: ", figname)
	plt.close(fig)


