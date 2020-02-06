# SRN
PyTorch Source code for "[3D Single-Person Concurrent Activity Detection Using Stacked Relation Network], AAAI2020

## Requirements
- python packages
  - pytorch = 0.4.1
  - torchvision>=0.2.1


## Data Preparation
 - Download the raw data from [UCLA Concurrent Activity Detection Dataset](data/UCLA_dataset/) and [UA Concurrent Activity Detection Dataset](data/UA-CAD). And pre-processes the data.

 - Preprocess the data with

    `src/data_proc_UCLA.py` or `src/data_proc_UA.py`



## Model Training
    `python train.py --checkpoint checkpoint/folder`

## Model Evaluation
    `python predict.py --checkpoint checkpoint/path`


## BibTeX
```
@article{yi2020concurrent,
  title={3D Single-Person Concurrent Activity Detection Using Stacked Relation Network},
  author={Wei, Yi and Li, Wenbo and Fan, Yanbo and Xu, Linghan and Chang, Ming-Ching and Lyu, Siwei},
  journal={The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI},
  year={2020}
}
```

## License
All materials in this repository are released under the MIT License.
