# Low-resource speaker recognition systems

This repository is used for researching the effectiveness of wav2vec2 
for speaker recognition in low-resource data conditions. 
See the (pre-print) paper at: ...

## Setting up 

### Dependencies

If poetry is not installed, see https://python-poetry.org/docs/. We also
expect at least python 3.8 on the system. If this is not the case, look into
https://github.com/pyenv/pyenv for an easy tool to install a specific
python version on your system. 

The python dependencies can be installed (in a project-specific virtual environment) by:

```bash
$ ./scripts/setup_dependencies.sh
```

To access the virtual environment, activate it with `poetry shell` or prepend `poetry run` to the command you want to run in the virtual environment.

### Environment

Copy the example environment variables:

```bash
$ cp .env.example .env 
```

Then fill in `.env` accordingly. 

## Data preparation

We we will work towards the following folder structure under `$DATA_FOLDER` specified in `.env`. 
Note: all the data (zip files, extracted files, data shards) will take approximate 1 TB of storage space.

```
${DATA_FOLDER}
├── archives -> ../voxceleb/
├── meta
│   ├── iden_split.txt
│   ├── list_test_all2.txt
│   ├── list_test_all.txt
│   ├── list_test_hard2.txt
│   ├── list_test_hard.txt
│   ├── veri_test2.txt
│   ├── veri_test.txt
│   ├── vox1_meta.csv
│   └── vox2_meta.csv
├── shards_eval
│   ├── extended
│   ├── hard
│   └── original
├── shards_vox2_full
├── shards_vox2_tiny_deep
├── shards_vox2_tiny_shallow
├── voxceleb1
│   ├── test
│   └── train
└── voxceleb2
    └── train

```

### Downloading voxceleb1 and voxceleb2

I've experienced that the download links for voxceleb1/2 can be unstable.
I recommend manually downloading the dataset from the google drive link displayed 
on https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html.

You should end up 4 zip files, which are to be placed in `$DATA_FOLDER/archives`. 

1. `vox1_dev_wav.zip` 
2. `vox1_test_wav.zip`
3. `vox2_dev_aac.zip`
4. `vox2_test_aac.zip`

### Converting voxceleb2 to wav format



### creating datasets

#### full voxceleb2

...

#### tiny-shallow (many speakers, few files)

...

#### tiny-deep (few speakers, many files)

...

#### evaluation set(s)

...

## Experiments 

### X-vector

#### Initial naive hyperparameter search grid:

```
python run.py -m \
+experiment=xvector_full,xvector_tiny_few,xvector_tiny_many_low,xvector_tiny_many_high \
optim.algo.lr=1e-7,1e-6,1e-5,1e-4,1e-3,1e-2 \
tag=naive_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### Follow-up detailed hyperparameter search grid:

##### tiny-few

```
python run.py -m \
+experiment=xvector_tiny_few \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### tiny-many-high

```
python run.py -m \
+experiment=xvector_tiny_many_high \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### tiny-many-low

```
python run.py -m \
+experiment=xvector_tiny_many_low \
optim.algo.lr=1.78e-5,3.16e-5,5.62e-5,1.78e-4,3.16e-4,5.62e-4 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### full voxceleb2

```
python run.py -m \
+experiment=xvector_full \
optim.algo.lr=1.78e-4,3.16e-4,5.62e-4,1.78e-3,3.16e-3,5.62e-3 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### Number of steps with best LR

##### tiny-few

```
python run.py -m \
+experiment=xvector_tiny_few \
optim.algo.lr=1.78E-02 \
tag=step_exp \
enroll_training_data=false post_process_scores=false \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=7348,12935,6622 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### tiny-many-high

```
python run.py -m \
+experiment=xvector_tiny_many_high \
optim.algo.lr=1.78E-03 \
tag=step_exp \
enroll_training_data=false post_process_scores=false \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=32188,60801,73677 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### tiny-many-low

```
python run.py -m \
+experiment=xvector_tiny_many_low \
optim.algo.lr=1.00E-04 \
tag=step_exp \
enroll_training_data=false post_process_scores=false \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=30921,90124,70956 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### full voxceleb2

```
python run.py -m \
+experiment=xvector_full \
optim.algo.lr=3.16E-03 \
tag=step_exp \
enroll_training_data=false post_process_scores=false \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=51412,38396,11558 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

### ECAPA-TDNN

Initial hyperparameter search:

```
python run.py -m \
+experiment=ecapa_full,ecapa_tiny_few,ecapa_tiny_many_low,ecapa_tiny_many_high \
optim.algo.lr=1e-7,1e-6,1e-5,1e-4,1e-3,1e-2 \
tag=naive_grid \
enroll_training_data=false post_process_scores=false \
data.use_additional_test_sets=true \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### Follow-up detailed hyperparameter search grid:

##### tiny-few

```
python run.py -m \
+experiment=ecapa_tiny_few \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### tiny-many-high

```
python run.py -m \
+experiment=ecapa_tiny_many_high \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### tiny-many-low

```
python run.py -m \
+experiment=ecapa_tiny_many_low \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

##### full voxceleb2

```
python run.py -m \
+experiment=ecapa_full \
optim.algo.lr=1.78e-3,3.16e-3,5.62e-3,1.78e-2,3.16e-2,5.62e-2 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm hydra.launcher.array_parallelism=3
```

#### Number of steps with best LR

##### tiny-few

```
python run.py -m \
+experiment=ecapa_tiny_few \
optim.algo.lr=1.78E-02 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=87510,83832,49529 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### tiny-many-high

```
python run.py -m \
+experiment=ecapa_tiny_many_high \
optim.algo.lr=5.62E-03 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=89540,80774,47997 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### tiny-many-low

```
python run.py -m \
+experiment=ecapa_tiny_many_low \
optim.algo.lr=5.62E-03 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=54190,54160,86823 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

##### full voxceleb2

```
python run.py -m \
+experiment=ecapa_full \
optim.algo.lr=5.62E-03 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=69317,43142,8754 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_11vram hydra.launcher.array_parallelism=1
```

### wav2vec2

Initial hyperparameter search:

```
python run.py -m \
+experiment=wav2vec2_full,wav2vec2_tiny_few,wav2vec2_tiny_many_high,wav2vec2_tiny_many_low \
optim.algo.lr=1e-7,1e-6,1e-5,1e-4,1e-3,1e-2 \
tag=naive_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=6
```

#### Follow-up detailed hyperparameter search grid:

##### tiny-few

```
python run.py -m \
+experiment=wav2vec2_tiny_few \
optim.algo.lr=1.78e-6,3.16e-6,5.62e-6,1.78e-5,3.16e-5,5.62e-5 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3
```

##### tiny-many-high

```
python run.py -m \
+experiment=wav2vec2_tiny_many_high \
optim.algo.lr=1.78e-5,3.16e-5,5.62e-5,1.78e-4,3.16e-4,5.62e-4 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3
```

##### tiny-many-low

```
python run.py -m \
+experiment=wav2vec2_tiny_many_low \
optim.algo.lr=1.78e-5,3.16e-5,5.62e-5,1.78e-4,3.16e-4,5.62e-4 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3
```

##### full voxceleb2

```
python run.py -m \
+experiment=wav2vec2_full \
optim.algo.lr=1.78e-5,3.16e-5,5.62e-5,1.78e-4,3.16e-4,5.62e-4 \
tag=detailed_grid \
enroll_training_data=false post_process_scores=false \
hydra/launcher=slurm_24vram hydra.launcher.array_parallelism=3
```

#### Number of steps with best LR

##### tiny-few

```
python run.py -m \
+experiment=wav2vec2_tiny_few \
optim.algo.lr=5.62E-06 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=62204,20427,456 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_24vram
```

##### tiny-many-high

```
python run.py -m \
+experiment=wav2vec2_tiny_many_high \
optim.algo.lr=1.782E-04 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=73232,95015,69018 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_24vram
```

##### tiny-many-low

```
python run.py -m \
+experiment=wav2vec2_tiny_many_low \
optim.algo.lr=1.782E-04 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=45979,43354,68179 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_24vram
```

##### full voxceleb2

```
python run.py -m \
+experiment=wav2vec2_full \
optim.algo.lr=1.782E-04 \
tag=step_exp \
enroll_training_data=true post_process_scores=true \
trainer.max_steps=25_000,50_000,100_000,400_000 \
seed=69317,43142,8754 \
data.use_additional_test_sets=true \
hydra/launcher=slurm_24vram
```

### Ablation

#### baseline

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
seed=787,23916 \
tag=baseline \
hydra/launcher=slurm_24vram
```

#### LR schedule

constant LR

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
optim/schedule=constant \
seed=11469,787,23916 \
tag=lr_constant \
hydra/launcher=slurm_24vram
```

exponentially decaying LR

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
optim/schedule=exp_decay \
seed=11469,787,23916 \
tag=lr_exp_decay \
hydra/launcher=slurm_24vram
```

Cyclic learning rate (but only 1 cycle), baseline has 4 cycles

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
++optim.schedule.scheduler.step_size_up=25_000 \
++optim.schedule.scheduler.step_size_down=25_000 \
seed=11469,787,23916 \
tag=lr_1_cycle \
hydra/launcher=slurm_24vram
```

#### Weights

Random initialization instead of pre-trained weights, nothing frozen

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
++network.reset_weights=true \
++network.completely_freeze_feature_extractor=false \
++network.wav2vec_initially_frozen=false \
seed=11469,787,23916 \
tag=weights_random_init \
hydra/launcher=slurm_24vram
```

Pretrained initialization, nothing frozen

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
++network.completely_freeze_feature_extractor=false \
++network.wav2vec_initially_frozen=false \
seed=11469,787,23916 \
tag=weights_no_freeze \
hydra/launcher=slurm_24vram
```

freeze 1st cycle

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
++network.completely_freeze_feature_extractor=false \
++network.wav2vec_initially_frozen=true \
++network.num_frozen_steps=12500 \
seed=11469,787,23916 \
tag=weights_freeze_cycle \
hydra/launcher=slurm_24vram
```

freeze CNN

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
++network.completely_freeze_feature_extractor=true \
++network.wav2vec_initially_frozen=false \
seed=11469,787,23916 \
tag=weights_freeze_cycle_cnn \
hydra/launcher=slurm_24vram
```



#### regularisation

No regularisation

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
network/regularisation=wav2vec2_none \
seed=11469,787,23916 \
tag=reg_none \
hydra/launcher=slurm_24vram
```

Only dropout

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
network/regularisation=wav2vec2_only_dropout \
seed=11469,787,23916 \
tag=reg_dropout \
hydra/launcher=slurm_24vram
```

Only masking

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
network/regularisation=wav2vec2_only_masking \
seed=11469,787,23916 \
tag=reg_mask \
hydra/launcher=slurm_24vram
```

Only layerdrop

```
python3 run.py -m \
+experiment=ablation_wav2vec2_tiny_few,ablation_wav2vec2_tiny_many_high \
network/regularisation=wav2vec2_only_layerdrop \
seed=11469,787,23916 \
tag=reg_layerdrop \
hydra/launcher=slurm_24vram
```