# ComplexHyperbolicKGE
Source code for the EMNLP22 paper [Complex Hyperbolic Knowledge Graph Embeddings with Fast Fourier Transform](https://arxiv.org/abs/2211.03635). 

This project is built upon the repository [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://github.com/HazyResearch/KGEmb), where the datasets WN18RR and FB15K can be found.

python version: 3.7.1

## Library Overview

This implementation includes the following complex hyperbolic KG embedding models:
 * FFTRotH, FFTRefH, FFTAttH


## Usage

To train and evaluate the models, run the following command:
```bash
zsh run_tuning_fft.sh
```
or equivalently 
```bash
echo $GPU"
"$DATASET"
"$MODEL"
"$REGULARIZER"
"$REG"
"$OPTIMIZER"
"$RANK"
"$BATCH_SIZE"
"$NEG_SAMPLE_SIZE"
"$LEARNING_RATE"
"$DOUBLE_NEG"
" \
    | xargs -L 11 -P $PARALLEL ./tuning_fft.sh
```
where
```
$GPU                                choose which cuda to use
$DATASET={WN18RR,FB15K}             dataset
$MODEL={FFTRotH,FFTRefH,FFTAttH}    embedding model
$REGULARIZER={N3,F2}                regularizer
$REG                                regularizer weight
$OPTIMIZER={Adagrad,Adam}           optimizer
$RANK                               embedding dimension + 1 (note that the rank for FFT models should be dimension + 1, e.g., $RANK=33 represents the 32-dimensional complex hyperbolic space for FFT models)
$BATCH_SIZE                         batch size
$NEG_SAMPLE_SIZE                    negative sample size
$LEARNIN_RATE                       learning rate
$DOUBLE_NEG                         {0,1}, whether to negative sample both head and tail entities
$PARALLEL                           the number of parallel programs
```
More parameters (e.g., epochs) can be referred to tuning_fft.sh.
The best parameters for WN18RR (tuned on $RANK=33):
```
$MODEL=FFTRotH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adam $RANK=33    $BATCH_SIZE=500 $NEG_SAMPLE_SIZE=100    $LEARNING_RATE=0.0003   $DOUBLE_NEG=1
$MODEL=FFTRefH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adam $RANK=33    $BATCH_SIZE=500 $NEG_SAMPLE_SIZE=100    $LEARNING_RATE=0.0003   $DOUBLE_NEG=1
$MODEL=FFTAttH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adam $RANK=33    $BATCH_SIZE=500 $NEG_SAMPLE_SIZE=100    $LEARNING_RATE=0.0004   $DOUBLE_NEG=1
```
The best parameters for FB237 (tuned on $RANK=33):
```
$MODEL=FFTRotH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adam    $RANK=33    $BATCH_SIZE=100 $NEG_SAMPLE_SIZE=100    $LEARNING_RATE=0.0002   $DOUBLE_NEG=0
$MODEL=FFTRefH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adagrad $RANK=33    $BATCH_SIZE=500 $NEG_SAMPLE_SIZE=250    $LEARNING_RATE=0.02     $DOUBLE_NEG=0
$MODEL=FFTAttH  $REGULARIZER=N3 $REG=0.0    $OPTIMIZER=Adagrad $RANK=33    $BATCH_SIZE=500 $NEG_SAMPLE_SIZE=100    $LEARNING_RATE=0.03     $DOUBLE_NEG=0
```

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--gpu] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {FFTRotH,FFTRefH,FFTAttH}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c] [--save_dir]
```

Description of arguments:
```bash
  -h, --help            show this help message and exit
  --gpu                 choose which cuda to use
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {FFTRotH,FFTRefH,FFTAttH}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
  --save_dir            Where to save the log and model
```

### Citation
If you find this repository useful for your research, please kindly cite our paper:
```angular2
@inproceedings{xiao2022emnlp,
  author    = {Huiru Xiao and
               Xin Liu and
               Yangqiu Song and
               Ginny Y. Wong and
               Simon See},
  title     = {Complex Hyperbolic Knowledge Graph Embeddings with Fast Fourier Transform},
  booktitle = {{EMNLP}},
  publisher = {Association for Computational Linguistics},
  year      = {2022}
}

```
