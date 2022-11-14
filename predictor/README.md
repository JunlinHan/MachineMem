## Training

### Preparation:

We use LaMem datasets for training. Please download [LaMem](http://memorability.csail.mit.edu/download.html).

We use pre-trained MoCo v2 as our initialization, please download it [here](https://drive.google.com/file/d/1zvDB_zBX_CwW5zumsVFSdldjw2eIYXik/view?usp=share_link).

Labels of both machine memorability scores and human memorability scores are provided in this repo.
### Commands:

Train a MachineMem predictor, run:
```
python main_predictor.py --inter_aug [path to lamem]
```

For HumanMem predictor, run:
```
python main_predictor.py --inter_aug --label_filename ./humanmem_scores.txt  [path to lamem]
```

Please note the folder with images should be a subfolder of your path folder.  

## Evaluation (prediction)

```
python main_predictor.py -e --resume [path to predictor]  [path to your dataset] 
```
Results will be saved in ./test_result.txt. 

If you only have few images to evaluate, you might try our demo at [project page](https://junlinhan.github.io/projects/machinemem.html) or [hugging face](https://huggingface.co/spaces/Junlinh/memorability_prediction).


## Pre-trained models

We provide our pre-trained MachineMem/HumanMem predictors at [this link](https://drive.google.com/drive/folders/1tO4ruBAToGSLZZ8VzAJ6v6O_99p6AJKI?usp=share_link).