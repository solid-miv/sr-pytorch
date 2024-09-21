# Super Resolution App

This is a Super Resolution App. It enables the user to upscale his photos by the factors of 2 and 4 with a help of **SRCNN**, **EDSR** and **SRGAN**. All the models were built with a `PyTorch` framework.

## How to run a project
1. Navigate to the project folder.
2. Set up a virtual environment. See `requirements.txt` for the versions.
3. Activate a virtual environment.
4. Run `python main.py` command.

## Notebooks
If you are interested in the models architecture are hyperparameters, then you can find all the notebooks in `notebooks/` directory. 

## Data description
All the models were trained on the `DIV2K` dataset available [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## Project structure
- `assets/` contains the icon and screenshots used in the project.
- `models/` contains 6 subfolders with the saved models and supplementary `.py`-files.
- `notebooks/` contains 6 Jupyter notebooks (one for each model).
- `main.py` that you can run to start the project.
- `requirements.txt` with external python modules used in the project and their versions.

## App preview

### SRCNN

![SRCNN 2x Preview](/assets/srcnn_2x_preview.png)

### EDSR
![EDSR 4x Preview](/assets/edsr_4x_preview.png)

### SRGAN
![SRGAN 4x Preview](/assets/srgan_4x_preview.png)