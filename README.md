## MoE-POT:Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training (NeurIPS 2025)

Code for [paper] MoEPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (NeurIPS'2025). It pretrains neural operator transformers (from **30M** to **0.5B**)  on multiple PDE datasets. Pre-trained weights could be found at here (I have also uploaded the corresponding training results for DPOT ).

![fig1](/resources/MoE-POT.png)

Our pre-trained MoE-POT achieves the state-of-the-art performance on multiple PDE datasets and could be used for finetuning on different types of downstream PDE problems.

![fig2](/resources/MoE-POT_result.png)

### Usage 

##### Pre-trained models

We have five pre-trained checkpoints of different sizes. Pre-trained weights are at here

| Size   | Attention dim | MLP dim | Layers | Heads | Model size |
| ------ | ------------- | ------- | ------ | ----- | ---------- |
| Tiny   | 512           | 512     | 4      | 4     | 30M        |
| Small  | 1024          | 1024    | 6      | 8     | 166M       |
| Medium | 1024          | 2038    | 8      | 8     | 489M       |


##### Dataset Protocol

All datasets are stored using hdf5 format, containing  `data`  field. Some datasets are stored with individual hdf5 files, others are stored within a single hdf5 file.

In `data_generation/preprocess.py`,  we have the script for preprocessing the datasets from each source. Download the original file from these sources and preprocess them to `/data` folder.

| Dataset       | Link                                                         |
| ------------- | ------------------------------------------------------------ |
| FNO data      | [Here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| PDEBench data | [Here](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) |
| PDEArena data | [Here](https://microsoft.github.io/pdearena/datadownload/)   |
| CFDbench data | [Here](https://cloud.tsinghua.edu.cn/d/435413b55dea434297d1/) |

In `utils/make_master_file.py` , we have all dataset configurations. When new datasets are merged, you should add a configuration dict. It stores all relative paths so that you could run on any places. 

##### Single GPU Pre-training

```python
python train_temporal.py
# or
python trainer.py --config_file ns2d_pretrain.yaml
```

##### Multiple GPU Pre-training

```python
python parallel_trainer.py --config_file pretrain_tiny.yaml
# tiny ,small ,medium three model
```

##### Configuration file

Now I use yaml as the configuration file. You could specify parameters for args. If you want to run multiple tasks, you could move parameters into the `tasks` ,

```yaml
model: MoEPOT
width: 512
tasks:
 lr: [0.001,0.0001]
 batch_size: [256, 32] 
```

This means that you start 2 tasks if you submit this configuration to `trainer.py`. 

##### Requirement

Install the following packages via conda-forge

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib scikit-learn scipy pandas h5py -c conda-forge
conda install timm einops tensorboard -c conda-forge
```

### Code Structure

- `README.md`
- `train_temporal.py`: main code of single GPU pre-training auto-regressive model 
- `trainer.py`: framework of auto scheduling training tasks for parameter tuning
- `parallel_trainer.py` framework of auto scheduling training tasks for Mutil-GPU
- `train_temporal_parallel.py`main code of Mutil-GPU pre-training auto-regressive model 
- `utils/`
  - `criterion.py`:  loss functions of relative error
  - `griddataset.py`: dataset of mixture of temporal uniform grid dataset
  - `make_master_file.py`: datasets config file
  - `normalizer`: normalization methods (#TODO: implement instance reversible norm)
  - `optimizer`: Adam/AdamW/Lamb optimizer supporting complex numbers
  - `utilities.py`: other auxiliary functions
- `configs/`: configuration files for pre-training or fine-tuning
- `models/`
  - `moepot.py`:        MoEPOT model
  - `MoE_conv.py`:      moe model
  - `fno.py`:          FNO with group normalization
  - `mlp.py`