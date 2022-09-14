## Install dependencies

(Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://www.anaconda.com/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3 with dependencies:
	```
	conda env create -f env.yml
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	conda activate tdt4290
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.

## Updating environment
Updates the environment when the env.yml file is changed:

    1. activate the environment (do this every time you open a new terminal and want to run code):
	```
	conda activate tdt4290
	```

	This install will update the environment. The prune option uninstalls dependencies which were removed from env.yml 

	2. Update the conda environment:
	```
	conda env update --file env.yml --prune
	```


## Troubleshooting 

You may encounter some errors:

These can be resolved with:
```
solution
```