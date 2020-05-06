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

## Updates
2020/5/6 We upload the models on Drive.

## Requirements

We recommend you to use Anaconda [BaiduYun Link](https://pan.baidu.com/s/1_J9INU-UpiT43qormibAuw)(passward:1y3v) or [Geogle Drive](https://drive.google.com/file/d/1H64lTpR3xzlSRfUxfZa4dOhAYZJcO7RU/view?usp=sharing) to manage your libraries.


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
   Prepare data follow COCO format or you can download our [IC15dataset-BAIDU](https://pan.baidu.com/s/1GbF0PnWDKw3qn2o2XgpB7Q) (passward:ect5) or [Geogle Drive](https://drive.google.com/file/d/1ZWRQWJwhydoCsqdNlX80y94cKQedUywO/view?usp=sharing), and unzip it in 
```bash
   datasets/.
```
#### step 2
You need to modify ```maskrcnn_benchmark/config/paths_catalog.py``` to point to the location where your dataset is stored.

#### step 3
Download [ResNet50 model (ImageNet)-BAIDU](https://pan.baidu.com/s/1nYePd4BgsBjhToeD2y1RbQ)(passward:edt8) or [ResNet50 model(ImageNet)-Drive](https://drive.google.com/file/d/1GZRktoRS4hoXmsCrucl3liLyMzl56WK7/view?usp=sharing) and put it in ```ContourNet/```. 

### Test IC15
#### Test with our [proposed model-BAIDU](https://pan.baidu.com/s/15xHgwUeMs-EYfHiBvNH0MQ)(password:g49o) or [proposed meodel-Drive](https://drive.google.com/drive/folders/10iJcEuR90tpkkyoIJ4Zq5r2xjwUWYYbc?usp=sharing)
Put the folder in 
```bash 
   output/.
```
You need to set the resolution to 1200x2000 in ```maskrcnn_benchmark/data/transformstransforms.py``` (line 50 to 52).
Then run
```bash 
   bash test_contour.sh
```
### Evaluate
Put bo.json to ic15_evaluate/, then run
```bash 
   cd ic15_evaluate
   conda deactivate
   pip2 install polygon2
   conda install zip
   python2 eval_ic15
```
#### Results on IC15
|        Model       	| precision 	| recall 	| F-measure 	|
|:------------------:	|:---------:	|:------:	|:---------:	|
|      Paper   	|    86.1   	|     87.6   	|    86.9   	|   
|  This implementation 	|    84.0   	|     90.1   	|    87.0   	| 

### Train our model on IC15
As mentioned in our paper, we only use offical training images to train our model, data augmentation includes random crop, rotate etc. There are 2 strategies to initialize the parameters in the backbone:1) use the [ResNet50 model (ImageNet)-BAIDU](https://pan.baidu.com/s/1nYePd4BgsBjhToeD2y1RbQ)(passward:edt8) or [ResNet50 model (ImageNet)-Drive](https://drive.google.com/file/d/1GZRktoRS4hoXmsCrucl3liLyMzl56WK7/view?usp=sharing), this is provided by [Yuliang](https://github.com/Yuliang-Liu/Box_Discretization_Network), which is ONLY an ImageNet Model With a few iterations on ic15 training data for a stable initialization.2) Use model only pre-trained on ImageNet(modify the WEIGHT to ```catalog://ImageNetPretrained/MSRA/R-50``` in ```config/r50_baseline.yaml```). In this repository, we use the first one to train the model on this dataset.
#### Step 1:
Run
```bash 
   bash train_contour.sh
```
#### Step 2:
   Change the ROTATE_PROB_TRAIN to 0.3 and ROTATE_DEGREE to 10 in ```config/r50_baseline.yaml``` (corresponding modification also needs to be done in ```maskrcnn_benchmark/data/transformstransforms.py``` from line 312 to 317), then finetune the model for more 10500 steps (lr starts from 2.5e-4 and dot 0.1 when step = [5k,10k]).

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
