<div align="center">
<h1> Multi-Transmotion:<br>  Pre-trained Model for Human Motion Prediction </h1>
<h3>Yang Gao, Po-Chien Luan, Alexandre Alahi
</h3>
<h4> <i> Annual Conference on Robot Learning (CoRL), München, November 2024 </i></h4>

[[Paper](https://arxiv.org/abs/2411.02673)] [[Website](https://www.epfl.ch/labs/vita/research/prediction/multi-transmotion/)] [[Poster](docs/Poster.pdf)] [[Slides](docs/CoRL_slides.pdf)]


<image src="docs/multi-transmotion.png" width="500">

</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

The ability of intelligent systems to predict human behaviors is crucial, particularly in fields such as autonomous vehicle navigation and social robotics. However, the complexity of human motion have prevented the development of a standardized dataset for human motion prediction, thereby hindering the establishment of pre-trained models. In this paper, we address these limitations by integrating multiple datasets, encompassing both trajectory and 3D pose keypoints, to propose a pre-trained model for human motion prediction. We merge seven distinct datasets across varying modalities and standardize their formats. To facilitate multimodal pre-training, we introduce Multi-Transmotion, an innovative transformer-based model designed for cross-modality pre-training. Additionally, we present a novel masking strategy to capture rich representations. Our methodology demonstrates competitive performance across various datasets on several downstream tasks, including trajectory prediction in the NBA and JTA datasets, as well as pose prediction in the AMASS and 3DPW datasets.
</br>


# Getting Started

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

# Unified Human Motion Data Framework

The framework is using modified trajnet++ to generate specific tabular format in '.ndjson' files. Specifically, the first part of '.ndjson' file represents trainable scenes/samples and the second part denotes the detailed human motion information. There are 207 columns of comprehensive human data are listed below:
- `[0]`: Frame ID
- `[1]`: Agents ID
- `[2:4]`: Trajectory (x, y)
- `[4:8]`: 3D Bounding Box (h, w, l, rot_z)
- `[8:12]`: 2D Bounding BOx (bb_left, bb_top, bb_width, bb_height)
- `[12:129]`: 3D pose keypoitns (x0,x2, ..x38, y0, y2, ..y38, z0, z2, ..z38)
- `[129:207]`: 2D pose keypoitns (xx0,xx2, ..xx38, yy0, yy2, ..yy38)
  
    The missing data will be denoted as 'null'.

Here we will show an example to convert the NBA dataset into the Unified Human Motion Dataframework. We have conveniently added the data of NBA to the release section of the repository (for license details, please refer to the original papers).

1. Download the data from our release and place the raw data subdirectory of NBA under `extract_data/NBA/raw/`
2. Extract NBA data
    ```
    python extract_data/NBA/extract_NBA.py --split train
    ```
3. Convert extracted data into UniHuMotion framework.
    ```
    mkdir -p UniHuMotion_trajnetpp/data/UniHuMotion_NBA/ data/UniHuMotion/
    mv extract_data/NBA/output_csv/nba_train.csv UniHuMotion_trajnetpp/data/UniHuMotion_NBA/
    cd UniHuMotion_trajnetpp
    python -m trajnetdataset.convert --acceptance 1.0 1.0 1.0 1.0 --train_fraction 1.0 --val_fraction 0.0 --fps 5 --obs_len 10 --pred_len 20 --chunk_stride 1
    ```
4. Generate trainable data cache (Efficient dataloading).
    ```
    mv output_pre UHM_NBA
    mv UHM_NBA ../data/UniHuMotion/
    cd ..
    python UniHuMotion_cache/cache_generator.py --UniHuMotion_dataset UHM_NBA
    ```
    
    
## Pre-training and fine-tuning
The code is using pytorch DDP so that it can be easily deployed on multiple gpus.

Demo command for pre-training:
```
python pretrain/train.py --cfg pretrain/configs/UniHuMotion.yaml --exp_name test_git
```

Demo command for fine-tuning on NBA dataset:
```
python ft_NBA/train.py --cfg ft_NBA/configs/UniHuMotion.yaml --exp_name default
```
Demo command for evaluation on NBA dataset:
```
python ft_NBA/evaluate.py --ckpt ./experiments/default_experiment/checkpoints/checkpoint_default/FT_NBA_ckpt.pth.tar --metric ade_fde
```

For the ease of use, we have also provided the pre-trained fine-tuned models in the release section of this repo.

# Work in Progress

This repository is work-in-progress and will continue to get updated and improved over the coming months.

```
@inproceedings{gao2024multi,
    title={Multi-Transmotion: Pre-trained Model for Human Motion Prediction},
    author={Gao, Yang and Luan, Po-Chien and Alahi, Alexandre},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024}
}
```

This work is a follow-up work of Social-Transmotion:
```
@InProceedings{saadatnejad2024socialtransmotion,
    title={Social-Transmotion: Promptable Human Trajectory Prediction}, 
    author={Saadatnejad, Saeed and Gao, Yang and Messaoud, Kaouther and Alahi, Alexandre},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```

---

## Reproduction notes and additional findings (fork by Santhosh335)

I reproduced this paper's NBA results on a Windows laptop with no GPU, using the provided pretrained checkpoint for inference only. Getting there took a bit of debugging — two compatibility bugs in the original pipeline, both fixed in this fork below. I also ran a small follow-up study on evaluation-set size, described further down.

### Reproduction result

Using `FT_NBA_ckpt.pth.tar`, evaluated on 1,500 clips of NBA test data:

| Metric | This reproduction | Paper (MinADE20 / MinFDE20) |
|---|---|---|
| ADE | 0.7442 | 0.75 |
| FDE | 0.9584 | 0.97 |

Close enough to the published numbers that I'm confident the pipeline is behaving correctly.

### Two bug fixes (Windows / Python 3.11)

The original code assumes a Linux environment and an older Python version, so running it on Windows with Python 3.11 surfaced two issues:

1. **`extract_data/NBA/extract_NBA.py`** — the CSV writer opens its output file without `newline=''`. On Windows this causes every row to be double-line-terminated (`\r\r\n`), which the downstream converter reads as a blank row inserted between every real row. This showed up as a fairly cryptic `IndexError: list index out of range` a couple of steps downstream — took a bit of tracing to find the actual cause. Fixed by adding `newline=''` to the `open()` call.
2. **`UniHuMotion_cache/utils/reader.py`** — `random.sample()` was called directly on a `dict_keys` view, which Python 3.9+ no longer allows (`TypeError: Population must be a sequence`). Fixed by wrapping it in `list(...)`.

### Extra: how much test data do you actually need?

I hit RAM limits on my laptop trying to convert the full ~32,500-clip test set (the conversion step loads everything into memory at once), so I ended up running on a reduced subset instead. That got me curious whether a smaller subset gives a meaningfully different ADE/FDE estimate, so I ran the same evaluation across 7 subset sizes:

| Test clips | ADE | FDE |
|---|---|---|
| 100 | 0.7316 | 0.9112 |
| 300 | 0.7566 | 0.9780 |
| 600 | 0.7531 | 0.9559 |
| 1,000 | 0.7439 | 0.9468 |
| 1,500 | 0.7442 | 0.9584 |
| 3,000 | 0.7542 | 0.9704 |
| 6,000 | 0.7532 | 0.9668 |

Across a 60x range in subset size, ADE stays within 0.7316-0.7566 and FDE within 0.9112-0.9780, with no obvious trend as the subset grows. Even 100 clips lands close to the paper's reported numbers. Basically, if you're compute-constrained like I was, a small subset seems to be enough to sanity-check this model - you probably don't need the full test set just to get a reasonable ADE/FDE reading.

The script used for this (`run_subset_study.ps1` + `set_test_size.py`) is included in this fork.
