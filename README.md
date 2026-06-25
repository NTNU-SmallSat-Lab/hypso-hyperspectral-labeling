# hypso-hyperspectral-labeling

GUI tool for labeling **HYPSO hyperspectral scenes** using **napari**.


## Features

- Open HYPSO `.nc` scenes
- Interactive pixel labeling in a napari GUI
- Save/load label masks (`.npy`)
- RGB band selection and quick presets
- Label propagation (Linear SVM + PCA)
- Optional quick links to (if matching metadata JSON exists): 
  - Google Earth
  - NASA Worldview

---

## Requirements

- **Python 3.10+**
- A working Python virtual environment (`venv`)
- Dependencies installed through this project (`pyproject.toml`)

---

## Installation (venv)

### 1) Clone the repository

    git clone https://github.com/NTNU-SmallSat-Lab/hypso-hyperspectral-labeling.git
    cd hypso-hyperspectral-labeling

### 2) Create and activate a virtual environment

#### Windows (PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1

#### Windows (Command Prompt)

    python -m venv .venv
    .venv\Scripts\activate.bat

#### macOS / Linux

    python -m venv .venv
    source .venv/bin/activate

### 3) Upgrade pip

    python -m pip install --upgrade pip

### 4) Install the project

Install from the repository root (the folder containing `pyproject.toml`):

    pip install .

If you are developing/editing the code yourself, use editable install instead:

    pip install -e .

If installation fails with `Failed to build gdal` / `error: failed-wheel-build-for-install`, install a matching **GDAL wheel** manually then run `pip install .` again.
Prebuilt GDAL wheels from the **cgohlke geospatial-wheels** GitHub releases page.

## Running the labeler

Run the GUI as a Python module (recommended):

    python -m hypso_labeler.main --nc data/nile_2026-02-17T08-47-24Z-l1a.nc --out out --cal 1a

### Example with more options

    python -m hypso_labeler.main --nc data/nile_2026-02-17T08-47-24Z-l1a.nc --out out --mission H2 --cal 1d

### Command-line arguments

- `--nc` (required)  
  Path to the HYPSO `.nc` scene file.

- `--out` (required)  
  Output directory where label files will be saved.

- `--cal` (optional)  
  Calibration level to load. Choices:
  - `1a`
  - `1b`
  - `1c`
  - `1d`
  - `2a`  
  Default: `1a`

- `--mission` (optional)  
  HYPSO mission. Choices:
  - `H1`
  - `H2`  
  Default: `H2`

- `--verbose` (optional)  
  Prints additional loading/debug info.

---

## Input and output

### Input

- A HYPSO NetCDF file (`.nc`) passed with `--nc`

### Output

The tool saves labels as a NumPy array (`.npy`) in the output directory:

    out/<scene_stem>_labels.npy

Example:

    out/nile_2026-02-17T08-47-24Z-l1a_labels.npy

---

## Optional metadata file (for map buttons)

The GUI can open the scene location in **Google Earth** and **NASA Worldview** if a matching metadata JSON file exists.

### Expected folder layout

    project_folder/
    ├─ data/
    │  └─ nile_2026-02-17T08-47-24Z-l1a.nc
    ├─ meta/
    │  └─ nile_2026-02-17T08-47-24Z-meta.json
    └─ out/

### Expected metadata fields (examples)

The metadata JSON should contain latitude/longitude fields (either naming style is supported):

- `latitude` / `longitude`
- or `target_latitude` / `target_longitude`

Optional (used for NASA Worldview date):
- `timestamp_acquired_string` (example: `"2026-02-17 08:47:24+0000"`)

---

## Label classes

Current built-in classes:

- `0` = unlabeled
- `1` = cloud
- `2` = land
- `3` = sea
- `4` = snow
- `5` = sand
- `6` = shallow_water
- `7` = man_made

---

