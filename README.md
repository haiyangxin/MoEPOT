## MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training (NeurIPS 2025)

Code for [paper] MoEPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (NeurIPS'2025). It pretrains neural operator transformers (from **30M** to **0.5B**)  on multiple PDE datasets. Pre-trained weights could be found at [here](https://huggingface.co/xhy2878/MoEPOT)  (I have also uploaded the corresponding training results for DPOT ).

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
| PDEBench data | [Here](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/pdebench_data_urls.csv) |
| PDEArena data | [Here](https://huggingface.co/pdearena/datasets)   |
| CFDbench data | [Here](https://huggingface.co/datasets/chen-yingfa/CFDBench) |

In `utils/make_master_file.py` , we have all dataset configurations. When new datasets are merged, you should add a configuration dict. It stores all relative paths so that you could run on any places. 

### Naming convention in the code

In the code, we refer to the datasets by a different identifier than the original datasets, see the following table for a mapping,For specific data processing, please refer to `data_generation/preprocess.py`:

|Code Identifier|Original dataset|
| ----------------|------------------------- |
|ns2d_fno_1e-5|NavierStokes_V1e-5_N1200_T20|
|ns2d_fno_1e-4|NavierStokes_V1e-4_N10000_T30|
|ns2d_fno_1e-3|NavierStokes_V1e-3_N5000_T50|
|ns2d_pdb_M1e-1_eta1e-2_zeta1e-2|2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5|
|ns2d_pdb_M1_eta1e-2_zeta1e-2|2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5|
|swe_pdb|2D_rdb_NA_NA.h5|
|dr_pdb|2D_diff-react_NA_NA.h5|
|cfdbench|CFDBench|
|ns2d_pda|NavierStokes-2D|
|ns2d_cond_pda|NavierStokes-2D-conditoned|

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
  - `normalizer`: normalization methods
  - `optimizer`: Adam/AdamW/Lamb optimizer supporting complex numbers
  - `utilities.py`: other auxiliary functions
- `configs/`: configuration files for pre-training or fine-tuning
- `models/`
  - `moepot.py`:        MoEPOT model
  - `MoE_conv.py`:      moe model
  - `fno.py`:          FNO with group normalization
  - `mlp.py`

  ### Acknowledgements
  We would like to express our gratitude to all collaborators, fellow students, and anonymous reviewers for their valuable assistance. Special thanks are extended to [Zhongkai Hao](https://haozhongkai.github.io/) and [Kuan Xu](http://staff.ustc.edu.cn/~kuanxu/) for their significant support.
  And we would like to thank the following open-source projects and research works:

  [DPOT](https://github.com/HaoZhongkai/DPOT) for model architecture

  [poseidon](https://github.com/camlab-ethz/poseidon) for dataset

  ### Citation
  If you use MoE-POT in your research, please use the following BibTeX entry.

  ```
  @misc{wang2025mixtureofexpertsoperatortransformerlargescale,
      title={Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training}, 
      author={Hong Wang and Haiyang Xin and Jie Wang and Xuanze Yang and Fei Zha and Huanshuo Dong and Yan Jiang},
      year={2025},
      eprint={2510.25803},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.25803}, 
}
```
