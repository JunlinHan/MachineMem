## Preparation:
Clone GANalyze via:
```
git clone https://github.com/LoreGoetschalckx/GANalyze.git
cd GANalyze
```

Download pre-trained generators:
``` 
cd pytorch; sh download_pretrained.sh
```

Add mmnet.py and our pre-trained MachineMem/HumanMem predictor (available at [this link](https://drive.google.com/drive/folders/1tO4ruBAToGSLZZ8VzAJ6v6O_99p6AJKI?usp=share_link)) to GANalyze/pytorch/assessors.


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

Modify your checkpoint path to change between HumanMem and MachineMem. Make sure it is consistent with the one you are using in mmnet.py.