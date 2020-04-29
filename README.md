# ContourNet: Taking a Further Step toward Accurate Arbitrary-shaped Scene Text Detection

This is a pytorch-based implementation for paper [ContourNet](https://arxiv.org/abs/2004.04940) (CVPR2020). ContourNet is a contour-based text detector which represents text region with a set of contour points. This repository is built on the pytorch [maskrcnn](https://github.com/facebookresearch/maskrcnn-benchmark).

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation
- [ ] Experiment on more datasets
- [ ] re-organize and clean the parameters

## Requirements

We recommend you to use Anaconda [BaiduYun Link](https://pan.baidu.com/s/1_J9INU-UpiT43qormibAuw)(passward:1y3v) to manage your libraries.


### Step-by-step install

```bash
  conda create --name ContourNet
  conda activate ContourNet
  conda install ipython
  pip install ninja yacs cython matplotlib tqdm scipy shapely networkx pandas
  conda install pytorch=1.0 torchvision=0.2 cudatoolkit=9.0 -c pytorch
  conda install -c menpo opencv
  export INSTALL_DIR=$PWDcd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install
  cd $INSTALL_DIR
  git clone https://github.com/wangyuxin87/ContourNet.git
  cd ContourNet
  python setup.py build develop
```

## Experiment on IC15 dataset
### Data preparing 
#### step 1
   Prepare data follow COCO format or you can download our [IC15dataset](https://pan.baidu.com/s/1GbF0PnWDKw3qn2o2XgpB7Q) (passward:ect5), and unzip it in 
```bash
   datasets/.
```
#### step 2
You need to modify ```maskrcnn_benchmark/config/paths_catalog.py``` to point to the location where your dataset is stored.

#### step 3
Download [ResNet50 model (ImageNet)](https://pan.baidu.com/s/1-42b3MRQ5T7t7SPC8fUb2g)(passward:5di1) and put it in ```ContourNet/```. This is ONLY an ImageNet Model With a few iterations on ic15 training data for a stable initialization provided by [Yuliang](https://github.com/Yuliang-Liu/Box_Discretization_Network).

### Test IC15
#### Test with our [proposed model](https://pan.baidu.com/s/15xHgwUeMs-EYfHiBvNH0MQ) (password:g49o)
Put the folder in 
```bash 
   output/.
```
You need to set the resolution to 1200x2000 in ```maskrcnn_benchmark/data/transformstransforms.py``` (line 48 to 51).
Then run
```bash 
   bash test_contour.sh
```
### Evaluate
Put bo.json to ic15_evaluate/, then run
```bash 
   cd ic15_evaluate
   conda deactivate
   pip2 install polygon 
   conda install zip
   python2 eval_ic15
```
#### Results on IC15
|        Model       	| precision 	| recall 	| F-measure 	|
|:------------------:	|:---------:	|:------:	|:---------:	|
|      Paper   	|    86.1   	|     87.6   	|    86.9   	|   
|  This implementation 	|    84.0   	|     90.1   	|    87.0   	| 

### Train our model on IC15
#### Step 1:
Run
```bash 
   bash train_contour.sh
```
#### Step 2:
   Change the ROTATE_PROB_TRAIN to 0.3 and ROTATE_DEGREE to 10 in ```config/ic15/r50_baseline.yaml``` (corresponding modification also needs to be done in ```maskrcnn_benchmark/data/transformstransforms.py``` from line 311 to 316), then finetune the model for more 10500 steps (lr starts from 2.5e-4 and dot 0.1 when step = [5k,10k]).

## Citation
If you find our method useful for your reserach, please cite
```bash 
  @article{wang2020contournet,
  title={ContourNet: Taking a Further Step toward Accurate Arbitrary-shaped Scene Text Detection},
  author={Wang, Yuxin and Xie, Hongtao and Zha, Zhengjun and Xing, Mengting and Fu, Zilong and Zhang, Yongdong},
  journal={CVPR},
  year={2020}
 ```
}

## Feedback
Suggestions and discussions are greatly welcome. Please contact the authors by sending email to ```wangyx58@mail.ustc.edu.cn```
