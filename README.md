## Synbuild-3D: NeurIPS 2025 Data and Code Samples

This repository contains code and instructions for working with the Synbuild-3D dataset, a large-scale dataset of semantically enriched 3D building models at Level of Detail 4.

### 📦 Dataset and Model Checkpoint Download

Download the code and data samples from [Google Drive folder](https://drive.google.com/drive/folders/1ULqxm23Bu3TNLg1ehze0ZWhCzPwfIlOA). 

 It contains:

- **`prod_run_5_2000.tar.gz`**  
  - A 33 GB compressed (262 GB uncompressed) dataset of 310,511 3D buildings and associated modalities used for training, evaluation, and dataset analysis.


- **`sample_100.zip`**  
  - A random sample of 100 3D buildings with their floor plans and segmentation masks — useful for quick visualization and inspection.


- **`MODEL_VAE_Point_Transformer_Coords_MLP_Adj_E_100_OPT_Adam_LR_0.0001.pt`**  
	- Baseline model checkpoint trained for 100 epochs on the `prod_run_5_2000` dataset (train/val/test splits in `/final_preprocessed_outdir`).


After downloading, place the files as follows:

```
Synbuild-3D-NeurIPS-Release/
├── baseline/
│   └── checkpoints/
│       └── MODEL_VAE_Point_Transformer_Coords_MLP_Adj_E_100_OPT_Adam_LR_0.0001.pt
├── data/
│   └── prod_run_5_2000/  # Extracted from prod_run_5_2000.tar.gz
│   └── sample_100/       # Extracted from sample_100.zip
```

### 🧪 Training and Visualization Setup

Set up your training environment with:

```bash
conda create -n synbuild-neurips python=3.11
conda activate synbuild-neurips
python -m pip install -r requirements.txt
```

#### ➤ Train the baseline model:

```bash
python baseline/train_baseline.py
```

### ➤ Visualize and analyze the dataset:

```bash
cd notebooks
jupyter notebook
```

Then open and run:

- `dataset_visualization.ipynb` — for 3D visualizations  
- `dataset_analysis.ipynb` — for dataset statistics

## 🧪 Evaluation setup following Graph Generation with Diffusion Mixture (GruM)

```bash
conda create -n synbuild-neurips-evaluation python=3.9.18
conda activate synbuild-neurips-evaluation
python -m pip install -r evaluation_requirements.txt
conda install -c conda-forge pyemd
conda install pyg -c pyg 
python -m ipykernel install --user --name synbuild-neurips-evaluation --display-name "Python (Synbuild-3D Evaluation)"
```

### ➤ Run evaluation:

```bash
cd notebooks
jupyter notebook
```

- Open validation_pipeline.ipynb
- Select the Python (Synbuild-3D Evaluation) kernel
- Run all cells to reproduce evaluation metrics