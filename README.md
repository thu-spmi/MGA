# MGA
This is the official code for the paper "[BUILDING MARKOVIAN GENERATIVE ARCHITECTURES OVER PRETRAINED LM BACKBONES FOR EFFICIENT TASK-ORIENTED DIALOG SYSTEMS](https://arxiv.org/abs/2204.06452)".
## Requirements
After you create an environment with `python 3.6`, the following commands are recommended to install required packages.
* pip install torch==1.5
* pip install transformers==3.5
* pip install spacy==3.1
* python -m spacy download en_core_web_sm
* pip install sklearn
* pip install tensorboard

Besides, you need to install the [standard evaluation repository](https://github.com/Tomiinek/MultiWOZ_Evaluation) for corpus-based evaluation, in which we change the references in `mwzeval/utils.py/load_references()` to 'damd', since we adopt the same delexicalization as [DAMD](https://github.com/thu-spmi/damd-multiwoz). 
## Data Preparation
The data preprocessing of MultiWOZ2.1 is based on the data preprocessing of [DAMD](https://github.com/thu-spmi/damd-multiwoz). For convenience, we provide preprocessed training data, database files and random generated testing goals in the format of `.zip` file. Execute following commands to unzip them
```
unzip db.zip -d ./db/
unzip ./data/multi-woz-2.1-processed/data.zip -d ./data/multi-woz-2.1-processed/
```
## Full Supervised Training
To train a MGA dialog system (GPT-2 backbone) in a supervised manner on the whole training set, run
```
bash shell_scripts/train.sh $GPU
```
You can change the settings in [shell_scripts/train.sh](shell_scripts/train.sh) (described in [config.py](config.py)) to run other architectures. For instance,
* Set `turn_level=False` and `only_target_loss=False` to run [UBAR](https://github.com/TonyNemo/UBAR-MultiWOZ) experiments.
* Set `turn_level=True` and `input_history=True` to run [SimpleTOD](https://github.com/salesforce/simpletod) experiments.

**Note**: you need to change the `exp_no` in [shell_scripts/train.sh](shell_scripts/train.sh) to prevent overwriting of different experiments.

If you want to train a MGA dialog system with T5 backbone, please refer to [this repository](https://github.com/cycrab/Mttod-for-mga), which is based on [the code of MTTOD](https://github.com/bepoetree/MTTOD).
## Semi-supervised Training
In the task of semi-supervised training, after trained with small amounts of labeled data, the models are further trained using both labeled and unlabeled data by variational learning (VL), as described in [VLS-GPT](https://arxiv.org/abs/2109.04314).

First, you need to pre-train a generative model and an inference model (both MGA) over the small amounts of labled data. For generative model training, run
```
bash shell_scripts/pretrain.sh $GPU $ratio
``` 
where `ratio` is the proportion of labeled data. For instance, you can train a generative model with only 20% labeled data by running
```
bash shell_scripts/pretrain.sh 0 20
``` 
For inference model training, run
```
bash shell_scripts/pretrain_infer.sh $GPU $ratio
```
After pretraining, you can conduct VL experiments, run
```
bash shell_scripts/train_VL.sh $GPU $ratio $generative_path $inference_path
```
Note that the `ratio` of pretrained generative model and inference model must be the same as the `ratio` in VL settings. For example, run a 20% semi-VL experiments with
```
path1="expriments_21/MGA-20/best_model"
path2="expriments_21/MGA-infer-20/best_model"
bash shell_scripts/train_VL.sh 0,1 20 $path1 $path2
```
If you want to run **session-level** VL experiments, set `turn_level=False` in [shell_scripts/pretrain.sh](shell_scripts/pretrain.sh), [shell_scripts/pretrain_infer.sh](shell_scripts/pretrain_infer.sh) and [shell_scripts/train_VL.sh](shell_scripts/train_VL.sh).
## Evaluation
To evaluate the trained dialog system (the generative model in VL experiment), run
```
bash shell_scripts/test.sh $GPU $path
```
where `path` is the checkpoint path of the model to be evaluated. Note that the settings in the [shell_scripts/test.sh](shell_scripts/test.sh) must be consistent with those during model training.




