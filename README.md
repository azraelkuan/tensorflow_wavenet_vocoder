# an implement of wavenet vocoder using tensorflow

**!!! the audio code is copied from [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) !!!**

**!!! the main tensorflow model is fixed from [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet/) !!!**

## Some issue
**mixture is in the branch of dev, but there are some bugs in generating wavs.**

## To Do
- [x] local condition
- [x] global condition
- [x] multi speaker
- [x] multi gpu
- [x] use conv1d transposed to upsample
- [ ] mixture logistic distribution (doing..)
- [ ] Tacotron + Wavenet Vocoder 

## Required
+ python >= 3.3
+ tensorflow > =1.3
+ tqdm
+ pyworld
+ pysptk 
+ nnmnkwii >= 0.12
+ scipy 
+ lws == 1.0

## Getting Start

### Download dataset
+ the voice conversion dataset(for multi speaker, 16k): [cmu_arctic](http://festvox.org/cmu_arctic/)
+ the single speaker dataset(22.05k): [LJSpeech-1.0](https://keithito.com/LJ-Speech-Dataset/)

### Preprocess data
for train faster, we should process the data to npy 
> `python preprocess.py --num_workers 4 --name ljspeech --in_dir /your_path/LJSpeech-1.0 --out_dir /your_outpath/ --hparams sample_rate=22050`

### Training
#### for single speaker
> `python train.py --num_gpus 4 --batch_size 2 --train_txt /your_train_txt/ --hparams gc_enable=False,global_channel=0,global_cardinality=0,NPY_DATAROOT=/your_npy_datadir/,sample_rate=22050 --logdir_root log_ljspeech`

#### for multi speaker
> `python train.py --batch_size 2 --num_gpus 4 --train_txt /your_train_txt/ --logdir_root log_arctic`

### Synthesize 
#### for single speaker
the eval_txt is extracted from the train_txt
>`python mul_generate.py --eval_txt /your_eval_txt/ --wav_out_path test_ljspeech.wav /your_cheakpoint/ ---hparams gc_enable=False,global_channel=0,global_cardinality=0,NPY_DATAROOT=/your_npy_datadir/,sample_rate=22050`

#### for multi speaker
> `python mul_generate.py --eval_txt /your_eval_txt/ --wav_out_path test_arctic.wav /your_checkpoint/ --gc_id 6`
