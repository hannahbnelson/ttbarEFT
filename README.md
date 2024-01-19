# ttbarEFT
Analysis repository for the dilepton ttbar and tW EFT analysis based on `coffea` and the `topcoffea` package. 

# Setup

If conda or micromamba are not already available, choose one and install: 
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
Or,
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Next, clone this repository: 
```
git clone git@github.com:hannahbnelson/ttbarEFT.git
cd ttbarEFT
```

Then, setup the conda environment: 
```
unset PYTHONPATH # To avoid conflicts.
conda env create -f environment.yml
conda activate ttbarEFT-env
pip install -e .
```

Alternatively, use micrcomamba: (not yet tested)
```
micromamba env create -f environment.yml
unset PYTHONPATH # To avoid conflicts.
micromamba activate ttbarEFT-env
pip install -e .
```

The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`. The `topcoffea` package upon which this analysis also depends is not yet available on `PyPI`, so we need to clone the `topcoffea` repo and install it ourselves.

```
cd /your/favorite/directory
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
pip install -e .  
```

Now all of the dependencies have been installed and the `ttbarEFT` repository is ready.
In the future, just activate the environment: 
```
unset PYTHONPATH
conda activate ttbarEFT-env
```

# Run a job using work\_queue
First, `cd` into `analysis`. Here you will find the `run_wq_processor.py` script. To run the script, do 
```
python run_processor.py --outname OutputName <PathToInputFile>
```
This creates the tasks, but you need to request some workers to execute them on distributed workers. 
Please note that the workers must be submitted from the same environment that you are running the run script from so open a new ssh session to `glados` and run these commands: 
```
unset PYTHONPATH
conda activate ttbarEFT-env
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores 12 --memory 48000 --disk 100000 10
```

The workers will terminate themselves after 15 minutes of inactivity. More details on the work queue executor can be found here.

A full example of the workflow is: 
```
ssh glados
unset PYTHONPATH
conda activate ttbarEFT-env
cd ttbarEFT/analysis
python run_wq_nanogen_processor.py --outname wq_test ../input_samples/sample_jsons/nanoGen_vast_TT1j2l_cQj31/

ssh glados
unset PYTHONPATH
conda activate ttbarEFT-env
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores 12 --memory 48000 --disk 100000 10
```

You can monitor the status of the workers with `work_queue_status`. 
