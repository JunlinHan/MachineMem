## Preparation:
Clone [GANalyze](https://github.com/LoreGoetschalckx/GANalyze) via:
```
git clone https://github.com/LoreGoetschalckx/GANalyze.git
cd GANalyze
```

Download pre-trained generators:
``` 
cd pytorch; sh download_pretrained.sh
```

Add mmnet.py and pre-trained MachineMem/HumanMem predictors (available at [this link](https://drive.google.com/drive/folders/1tO4ruBAToGSLZZ8VzAJ6v6O_99p6AJKI?usp=share_link)) to GANalyze/pytorch/assessors/.


## Training:
For MachineMem GANalyze:
``` 
python train_pytorch.py --assessor mmnet
```

For HumanMem GANalyze:
Simply change machinemem_predictor to humanmem_predictor in mmnet.py (line 29). 

``` 
python train_pytorch.py --assessor mmnet
```

## Testing:
Run:
``` 
python test_pytorch.py --checkpoint_dir [path to your checkpoint folder]
```

Modify your checkpoint path to swap between HumanMem and MachineMem. Make sure the checkpoint is consistent with the weights you are using in mmnet.py (machine or human).


## Cite:
If you use something from GANalyze, you may cite:
``` 
@inproceedings{goetschalckx2019ganalyze,
  title={Ganalyze: Toward visual definitions of cognitive image properties},
  author={Goetschalckx, Lore and Andonian, Alex and Oliva, Aude and Isola, Phillip},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5744--5753},
  year={2019}
}
``` 
