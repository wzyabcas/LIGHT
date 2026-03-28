<div align="center">

# **Unleashing Guidance Without Classifiers for Human-Object Interaction Animation**

[Ziyin Wang](https://github.com/wzyabcas/LIGHT)<sup>1</sup>&emsp; [Sirui Xu](https://sirui-xu.github.io)<sup>1</sup>&emsp; [Chuan Guo](https://ericguo5513.github.io/)<sup>2</sup>&emsp; [Bing Zhou](https://zhoubinwy.github.io/)<sup>2</sup>&emsp; [Jiangshan Gong](https://github.com/gong208)<sup>1</sup>&emsp; [Jian Wang](https://jianwang-cmu.github.io/)<sup>2</sup>&emsp; [Yu-Xiong Wang](https://yxw.cs.illinois.edu/)<sup>1</sup>&emsp; [Liang-Yan Gui](https://lgui.web.illinois.edu/)<sup>1</sup>

<sup>1</sup>University of Illinois Urbana-Champaign<br>
<sup>2</sup>Snap Inc.<br>

<sup>*</sup>Equal contribution

**ICLR 2026**

</div>

</p>
<p align="center">
  <a href='https://arxiv.org/pdf/2603.25734'>
    <img src='https://img.shields.io/badge/Arxiv-2509.09555-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <!-- <a href='https://arxiv.org/pdf/xxxx.xxxxx.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> -->
<a href='https://ziyinwang1.github.io/LIGHT/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a> 
  <a href='https://github.com/wzyabcas/LIGHT'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>


![](./assets/teaser.png)

## News
- [2026-03-26] Initial release of LIGHT.
- [2026-03-26] Release the inference pipeline.




## TODO
- [ ] Release the evaluation pipeline
- [ ] Release the training pipeline
- [ ] Release the augmentated data


## General Description

  We introduce LIGHT, a pipeline that generates realistic human-object interaction animations by denoising different components of the motion at different speeds, so cleaner components naturally guide noisier ones - producing contact-aware guidance without any external classifiers or hand-crafted priors.


## Preparation

<details>
  <summary>Please follow these steps to get started</summary>
1. Download SMPL+H and SMPL-X.

    Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in the AMASS project), DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL), and SMPL-X model from [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php). Then, please place all the models under `./models/`. The `./models/` folder tree should be:

    ```
    models
    │── smplh
    │   ├── female
    │   │   ├── model.npz
    │   ├── male
    │   │   ├── model.npz
    │   ├── neutral
    │   │   ├── model.npz
    │   ├── SMPLH_FEMALE.pkl
    │   ├── SMPLH_MALE.pkl
    │   └── SMPLH_NEUTRAL.pkl    
    └── smplx
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_FEMALE.pkl
        ├── SMPLX_MALE.npz
        ├── SMPLX_MALE.pkl
        ├── SMPLX_NEUTRAL.npz
        └── SMPLX_NEUTRAL.pkl
    ```

    Please follow [smplx tools](https://github.com/vchoutas/smplx/blob/main/tools/README.md#merging-smpl-h-and-mano-parameters) to merge SMPL-H and MANO parameters.

2. Prepare Environment

  - Create and activate a fresh environment:
    ```bash
    conda create -n interact python=3.8
    conda activate interact
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    ```

    To install PyTorch3D, please follow the official instructions: [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

    Install remaining packages:
    ```
    pip install -r requirements.txt
    ```

3. Prepare data

  - **OMOMO**

    Download the dataset from this [link](https://github.com/lijiaman/omomo_release)

    Expected File Structure:
    ```bash
    InterAct/omomo/sequences_canonical
    ├── objects
    │   └── object_name
    │       └── object_name.obj
    └── sequences
    	└── id
    		├── data.npz
    ```
</details>

## Inference

<details>
   <summary>Prepare</summary>

  Download pretrained model and evaluator models:

  -  Download the checkpoints of the pretrained evaluator and text encoder used in training from this [link](https://drive.google.com/file/d/1-bpafRyaVHdX4TsltDHiGIxcjw-k1Fnf/view?usp=sharing), and put in `./text2interaction/assets/eval`.

  -  Optional: Download the pretrained model checkpoints from this [link](https://drive.google.com/file/d/1vfskohWxr7gBuve1MLD1RlGut_xSN8mL/view?usp=sharing), and put in `./text2interaction/save/`.

  </details>

  <details>
  <summary>Evaluation</summary>

  To evaluate on our benchmark, execute the following steps

  - Evaluate on the marker representation:

    ```
    cd text2interaction
    bash ./scripts/eval.sh
    ```
  - Evaluate on the marker representation with contact guidance used:

    ```
    cd text2interaction
    bash ./scripts/eval_wguide.sh
    ```
    </details>

    


## Visualization

To visualize the dataset, execute the following steps:

1. Run the visualization script:

    ```bash
    python visualization/visualize.py [dataset_name]
    ```

    Replace [dataset_name] with one of the following: behave, neuraldome, intercap, omomo, grab, imhd, chairs.

2. To visualize markers, run:

    ```bash
    python visualization/visualize_markers.py
    ```



## Citation  

If you find this repository useful for your work, please cite:

```bibtex
@inproceedings{wang2026unleashing,
      title = {Unleashing Guidance Without Classifiers for Human-Object Interaction Animation},
      author = {Wang, Ziyin and Xu, Sirui and Guo, Chuan and Zhou, Bing and Gong, Jiangshan and Wang, Jian and Wang, Yu-Xiong and Gui, Liang-Yan},
      booktitle = {ICLR},
      year = {2026}
    }
```

Please also consider citing the InterAct benchmark that we built our model upon:
```bibtex
@inproceedings{xu2025interact,
    title     = {{InterAct}: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation},
    author    = {Xu, Sirui and Li, Dongting and Zhang, Yucheng and Xu, Xiyan and Long, Qi and Wang, Ziyin and Lu, Yunzhi and Dong, Shuchang and Jiang, Hezi and Gupta, Akshat and Wang, Yu-Xiong and Gui, Liang-Yan},
    booktitle = {CVPR},
    year      = {2025}}

```
