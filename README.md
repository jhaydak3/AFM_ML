# AFM Contact Point and Quality Classification Pipeline

## Overview

This project/codebase is designed to:

1. **Enable automatic identification of the contact point** in Atomic Force Microscopy (AFM) experiments using a neural network trained on annotated data.
2. **Identify low-quality AFM curves** using a neural network binary classifier.

By combining these two capabilities, the pipeline facilitates **high-content and high-throughput analysis of AFM elastographs**.

## Supported Data Formats

Currently, this pipeline is optimized for **Asylum Research Data Format (ARDF)** data converted to **HDF5 (.h5)** files. Other AFM data formats will require some preprocessing to be compatible. Ultimately, all processed data is stored in **MATLAB (.mat)** files for analysis. See an **example .mat file** for required fields. Feel free to reach out to me if you're using another brand of AFM.

## Repository Contents

This repository includes **processed .mat files** containing AFM data, along with scripts for **preprocessing, training models, and manual annotation**.

## Usage

### Preprocessing and Training

1. **Use the preprocessing scripts** in the `training` folder to generate preprocessed data.
2. **Use the training scripts** in the `training` folder to train:
   - The **regression model** for contact point identification.
   - The **binary classifier** for filtering low-quality curves.

These trained models can then be used together for **high-content AFM analysis**.

### Annotating Your Own Data

1. **Convert raw AFM data to .mat files**:
   - If starting from `.ARDF` files, convert them to `.h5` using the `ARDF.exe` converter located in the `AFM_data` folder.
2. **Run the `AFM_INPUT` script** located in the `GUI` folder.
3. **Use the graphical user interface (GUI) application** `app1_v4.mlapp` to manually annotate the data.

## Support

- For **issues, questions, or concerns**, contact **[jonathan.haydak@icahn.mssm.edu](mailto:jonathan.haydak@icahn.mssm.edu)**.
- For **complaints**, contact **[evren.azeloglu@mssm.edu](mailto:evren.azeloglu@mssm.edu)**.
