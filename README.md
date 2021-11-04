# CPEE
A Multi-Event Extraction Model on Nursing Records, pipeline method
images/image.png

# Background

Event extraction on Chinese nursing records.

We propose event schemas for nursing records and annotate a Chinese nursing event extraction dataset (CNEED) on ICU nursing records.  It is not convenient to open source, but some cases are provided in `data`. You can replace the dataset.

# Model
<img src="https://gitee.com/songruoyu/pic-bed/raw/master/img/202111032143539.png" alt="image-20211103214339491" width="600" height="400" />

# Usage

```
git clone https://github.com/ruoyusong/CPEE
```

# Requirements

```
pytorch-pretrained-bert==0.6.2
```

# Run

1. train Trigger Extractor

```
CUDA_VISIBLE_DEVICES=0 python dev.py --task_type trigger --learning_rate 5e-5
```

2. train Argument Extractor

```
CUDA_VISIBLE_DEVICES=0 python dev.py --task_type argument --learning_rate 3e-5
```

3. predict event results on test set

```
CUDA_VISIBLE_DEVICES=0 python predict.py
```

4. test event results that prediced in step 3

```
CUDA_VISIBLE_DEVICES=0 python test.py
```

**Finally, you can see the test results on `logs/test.log`**
