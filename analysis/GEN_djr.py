#!/usr/bin/env python3
# This script was originally from https://github.com/FNALLPC/cmseft2023/blob/main/generation/djr.py
# and then modified by Brent. 

import uproot
import os
import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import warnings

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

plt.style.use(hep.style.CMS)

if __name__ == '__main__':
    '''
    good example:
    root://cmseos.fnal.gov//store/user/dspitzba/EFT/qcut30.root

    bad example:
    root://cmseos.fnal.gov//store/user/cmsdas/2023/short_exercises/Generators/wjets_2j/w2jets_qcut10.root
    '''

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='root://cmseos.fnal.gov//store/user/dspitzba/EFT/qcut80.root', help="Input file")
    argParser.add_argument('--output', action='store', default='./djr.pdf', help="Output file")
    args = argParser.parse_args()

    djr_axis = hist.axis.Regular(40, -0.5, 3.5, name="djr", label=r"$\Delta JR$")
    parton_axis = hist.axis.Integer(0, 4, name="n", label="Number of partons")
    transition_axis = hist.axis.Integer(0, 6, name="t", label="DJR X->Y")
    djr = hist.Hist(djr_axis, parton_axis, transition_axis)

    dir_name = args.input
    files = [f for f in os.listdir(args.input) if '.root' in f]

    for fin in files:
        print(f"Loading input file {args.input}/{fin}")
        # Load in events from root file
        ar = events["GenEventInfoProduct_generator__GEN./GenEventInfoProduct_generator__GEN.obj"].arrays()
        djr_values = ar['GenEventInfoProduct_generator__GEN.obj']['DJRValues_']
        nMEPartons = ar['GenEventInfoProduct_generator__GEN.obj']['nMEPartons_']

        djr.fill(
            djr = np.log10(djr_values),
            n = ak.ones_like(djr_values)*nMEPartons,
            t = ak.local_index(djr_values),
        )

    print(f"Plotting...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, axs = plt.subplots(3,2, figsize=(15,21))

        for i in range(3):
            for j in range(2):
                transition = 2*i+j
                djr[:, :, transition].plot1d(
                        overlay='n',
                        ax=axs[i][j],
                        label= [f'{k} partons' for k in range(4)]
                )
                djr[:, :, transition][{'n':sum}].plot1d(
                        ax=axs[i][j],
                        label = ['total'],
                        color = 'gray',
                )

                axs[i][j].set_xlabel(r'$DJR\ %s \to %s$'%(transition, transition+1))
                axs[i][j].set_yscale('log')
                axs[i][j].legend(
                        loc='upper right',
                        bbox_to_anchor=(0.03, 0.88, 0.90, .11),
                        mode="expand",
                        ncol=2,
                )

        fig.savefig(args.output)
        print(f"Figure saved in {args.output}")
