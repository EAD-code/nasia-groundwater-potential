# Groundwater potential mapping in the Nasia sub-basin, northern Ghana

Python workflow comparing a **Frequency Ratio (FR)** model, a **Random Forest (RF)** model and a **hybrid FR–RF** model for groundwater potential mapping in the Nasia sub-basin (~5,300 km²), Voltaian Basin, northern Ghana. The models are validated against independent pumping-tested boreholes.

This repository holds the code that accompanies the manuscript:

> [Authors] ([Year]). *Developing a groundwater potential map in a data-scarce sedimentary basin using a Frequency Ratio–Random Forest comparison: the case of the Nasia sub-basin, Voltaian Basin, northern Ghana.* [Journal], [volume], [pages]. https://doi.org/[DOI]

## Contents

| File | Description |
|---|---|
| `nasia_fr_rf_v4_6.py` | Full workflow as a single Python script (version 4.6) |
| `Nasia_FR_RF_v4_6.ipynb` | The same workflow as a Jupyter notebook, one block per cell |

## What the workflow does

1. **Builds the conditioning factors** from raw data on one 30 m grid in WGS 84 / UTM Zone 30N (EPSG:32630). These are lithology, lineament density, distance to lineament, elevation, slope, TWI, TPI, distance to stream, rainfall, soil type and land use/land cover.
2. **Reads and labels the boreholes.** A well is labelled productive if its yield is above 0.6 m³/h (10 L/min). The `Role` column in the inventory separates the training wells (inventory) from the validation wells (pumping-tested). Wells outside the basin are dropped, and wells that fall on the same 30 m pixel are counted once.
3. **Screens the factors** for collinearity using Pearson correlation and VIF.
4. **Fits three models:**
   - **FR**, with area-proportional smoothing. Rare categorical classes (<1% of the basin) are merged.
   - **RF** on the raw factor values.
   - **Hybrid FR–RF**, an RF trained on the FR class values.
5. **Validates the models** on the independent wells, using ROC/AUC with bootstrap confidence intervals and DeLong tests. It also checks spatially separated wells (≥1 km from any training well), other yield thresholds (13 and 22 L/min) and out-of-bag AUC.
6. **Estimates uncertainty** with 50 bootstrap repetitions of the training wells. The mean of these repetitions is the **Groundwater Potential Index (GPI)**. The workflow also produces a per-pixel standard deviation and a stability layer.
7. **Produces the final groundwater potential map** by dividing the GPI into five equal-area zones, with a smoothed version for display.
8. **Runs independent checks**:
   - specific capacity and transmissivity at the validation wells;
   - regolith thickness, static water level, borehole depth and first water strike;
   - chloride mass balance recharge;
   - formation agreement between the inventory and the geological map.
9. **Shows which factors drive the result** with permutation importance and a leave-one-factor-out test.
10. **Writes all outputs:** manuscript figures, map layouts, GeoTIFFs, one Excel workbook with every table, a reproducibility record, and a report of anything missing or doubtful.

## Requirements

Python 3.10 or later. The easiest route is a conda environment:

```bash
conda create -n nasia -c conda-forge python=3.11 numpy pandas scipy scikit-learn matplotlib rasterio geopandas shapely pyproj jenkspy openpyxl pysheds jupyterlab
conda activate nasia
```

If a required library is missing, the script stops at the start and prints the install command.

## Input data

The input data are **not included** in this repository (see *Data availability*). The script expects the following files in one folder:

| Input | Expected file name(s) |
|---|---|
| Basin outline | `Nas_ia.shp` |
| DEM (30 m) | `Nasia_DEM.tif` or `NasiaDEM_1.tif` |
| Lineaments | `Nasia_Lineaments.shp` |
| Mapped rivers | `Nasia_rivers.shp` |
| Geology | `Nasia_Geology.shp` |
| Soil | `Soil.tif` |
| Land use / land cover | `landuse.tif` |
| Rainfall | `Annual_Rainfall.tif` (other candidates are tested automatically) |
| Borehole inventory | `Nasia_borehole_inventory_clean.csv` (columns `BH ID`, `Easting`, `Northing`, `Discharge(l/min)`, `Role`, plus hydraulic columns) |
| Recharge | `Nasia_CMB_recharge.csv` |

File names, column names and all settings are listed in **BLOCK 1**, the only block that needs editing. If your file names differ, change them there.

## How to run

1. Put the input files in one folder.
2. Open the script or notebook and, in **BLOCK 1**, set:
   ```python
   DATA_DIR   = r"path/to/your/data"
   OUTPUT_DIR = r"path/to/your/data/Nasia_outputs_v4"
   ```
3. Choose the run mode:
   ```python
   FAST_MODE = True    # quick test: 15 bootstrap repetitions
   FAST_MODE = False   # full run used in the paper: 50 repetitions
   ```
4. Run all cells in JupyterLab, or from the command line:
   ```bash
   python nasia_fr_rf_v4_6.py
   ```

A full run (`FAST_MODE = False`) takes from several minutes to about an hour, depending on the computer.

## Outputs

All outputs are written to `OUTPUT_DIR`:

| Folder | Contents |
|---|---|
| `figures/` | Manuscript figures (ROC curves, importance, hydraulic checks, etc.) |
| `final_maps/` | Map layouts of every input, factor, model surface, the GPI, potential zones, uncertainty and stability |
| `tables/` | Every table as CSV, plus one combined Excel workbook |
| `logs/` | Reproducibility record (settings, software versions, file checksums) and the missing-items report |

## Key settings used in the paper

| Setting | Value |
|---|---|
| Grid | 30 m, EPSG:32630 |
| Productive threshold | > 0.6 m³/h (10 L/min) |
| Training / validation wells | 152 / 58 (after removing wells outside the basin and duplicates on one pixel) |
| RF | 500 trees, randomised hyperparameter search |
| Bootstrap repetitions | 50 |
| Stream initiation area | 1 km² |
| Lineament density window | automatic (3,750 m radius in this dataset) |
| Potential zones | 5 equal-area classes of the GPI |
| Random seed | 42 |

## Data availability

The borehole records were obtained from [data providers] and cannot be redistributed. The remaining spatial layers came from the sources cited in the manuscript (Table 1). Requests for data should be sent to the corresponding author.

## Citation

If you use this code, please cite the paper above and this repository:

> [Authors] ([Year]). Nasia groundwater potential mapping code (v4.6). Zenodo. https://doi.org/[Zenodo DOI]

## Licence

MIT. See `LICENSE`.

## Contact

Elikplim Abla Dzikunoo, Department of Earth Science, University of Ghana, Legon, Accra. [email]
