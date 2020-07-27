# ContourNet: Taking a Further Step toward Accurate Arbitrary-shaped Scene Text Detection

This is a pytorch-based implementation for paper [ContourNet](https://arxiv.org/abs/2004.04940) (CVPR2020). ContourNet is a contour-based text detector which represents text region with a set of contour points. This repository is built on the pytorch [maskrcnn](https://github.com/facebookresearch/maskrcnn-benchmark).

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation
- [x] Experiment on more datasets
- [ ] re-organize and clean the parameters

## Updates
```bash
2020/5/6 We upload the models on Drive.
2020/6/11 We update the experiment for CTW-1500 and further detail some training settings.
```
## Requirements

We recommend you to use Anaconda [BaiduYun](https://pan.baidu.com/s/1_J9INU-UpiT43qormibAuw)(passward:1y3v) or [Drive](https://drive.google.com/file/d/1H64lTpR3xzlSRfUxfZa4dOhAYZJcO7RU/view?usp=sharing) to manage your libraries.


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
## Results
![image](https://github.com/wangyuxin87/ContourNet/blob/master/demo/display.png)

We use only official training images to train our model.

|        Dataset       	|        Model       	| recall 	| precision 	| F-measure 	|
|:------------------: |:------------------:	|:---------:	|:------:	|:---------:	|
|        [ic15](https://rrc.cvc.uab.es/?ch=4)       	|      Paper   	|    86.1   	|     87.6   	|    86.9   	|   
|        [ic15](https://rrc.cvc.uab.es/?ch=4)       	|  This implementation 	|    84.0   	|     90.1   	|    87.0   	| 
|  [CTW-1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) |      Paper   	|    84.1   	|     83.7   	|    83.9   	|   
|  [CTW-1500](https://github.com/Yuliang-Liu/Curve-Text-Detector) |  This implementation 	|    84.0   	|     85.7   	|    84.8   	| 

## Experiment on IC15 dataset
### Data preparing 
#### step 1
   Prepare data follow COCO format or you can download our IC15dataset [BAIDU](https://pan.baidu.com/s/1GbF0PnWDKw3qn2o2XgpB7Q) (passward:ect5) or [Geogle Drive](https://drive.google.com/file/d/1ZWRQWJwhydoCsqdNlX80y94cKQedUywO/view?usp=sharing), and unzip it in 
```bash
   datasets/.
```
#### step 2
You need to modify ```maskrcnn_benchmark/config/paths_catalog.py``` to point to the location where your dataset is stored.

#### step 3
Download ResNet50 model [BAIDU](https://pan.baidu.com/s/1nYePd4BgsBjhToeD2y1RbQ)(passward:edt8) or [Drive](https://drive.google.com/file/d/1GZRktoRS4hoXmsCrucl3liLyMzl56WK7/view?usp=sharing) and put it in ```ContourNet/```. 

### Test IC15
#### Test with our proposed model [BAIDU](https://pan.baidu.com/s/15xHgwUeMs-EYfHiBvNH0MQ)(password:g49o) or [Drive](https://drive.google.com/drive/folders/10iJcEuR90tpkkyoIJ4Zq5r2xjwUWYYbc?usp=sharing)
Put the folder in 
```bash 
   output/.
```
Set the resolution to 1200x2000 in ```maskrcnn_benchmark/data/transformstransforms.py``` (line 50 to 52). You can ignore this step when you train your own model, which seems to obtain better results. Then run
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

### Train our model on IC15
As mentioned in our paper, we only use offical training images to train our model, data augmentation includes random crop, rotate etc. There are 2 strategies to initialize the parameters in the backbone:1) use the ResNet50 model (ImageNet)[BAIDU](https://pan.baidu.com/s/1nYePd4BgsBjhToeD2y1RbQ)(passward:edt8) or [Drive](https://drive.google.com/file/d/1GZRktoRS4hoXmsCrucl3liLyMzl56WK7/view?usp=sharing), this is provided by [Yuliang](https://github.com/Yuliang-Liu/Box_Discretization_Network), which is ONLY an ImageNet Model With a few iterations on ic15 training data for a stable initialization.2) Use model only pre-trained on ImageNet(modify the WEIGHT to ```catalog://ImageNetPretrained/MSRA/R-50``` in ```config/ic15/r50_baseline.yaml```). In this repository, we use the first one to train the model on this dataset.
#### Step 1:
Run
```bash 
   bash train_contour.sh
```
#### Step 2:
   Change the ROTATE_PROB_TRAIN to 0.3 and ROTATE_DEGREE to 10 in ```config/ic15/r50_baseline.yaml``` (corresponding modification also needs to be done in ```maskrcnn_benchmark/data/transformstransforms.py``` from line 312 to 317), then finetune the model for more 10500 steps (lr starts from 2.5e-4 and dot 0.1 when step = [5k,10k](optional)).

## Experiment on CTW dataset
### Data preparing 
#### step 1
   Prepare data follow COCO format or you can download our [CTW-dataset](https://drive.google.com/file/d/1YbohYSs4T6yyVMEYCpr18fzKiUWzYVOe/view?usp=sharing), and unzip it in
```bash 
   output/.
```
#### step 2
   You need to modify ```maskrcnn_benchmark/config/paths_catalog.py``` to point to the location where your dataset is stored.
### Test CTW
#### Test with our proposed model [Drive](https://drive.google.com/drive/folders/1vEaYiS7Qxvhj6rdqTOATT-ke86FqGHnF?usp=sharing)
Put the folder in 
```bash 
   output/.
```  
Then run
```bash 
   bash test_contour.sh
```
### Evaluate
Run
```bash 
   cd ctw_eval
   python eval_ctw1500.py
```

### Train our model on CTW
Run
```bash 
   bash train_contour.sh
```
# Improvement
1. We use different reconstruction algorithm to rebuild text region from contour points for curved text, you can reproduce our approach used in the paper by modifying the hyper-parameter in Alpha-Shape Algorithm (some tricks also should be added). Furthermore, more robust reconstruction algorithm may obtain better results.
2. The detection results are not accurate when the proposal contains more than one text, because of that the strong response will be obtained in both contour regions of texts. 
3. Some morphological algorithms can make the contour line more smooth.
4. More tricks like deformable_conv, deformable_pooling in the box_head, etc. can further improve the detection results.

## Citation
If you find our method useful for your reserach, please cite
```bash 
 @inproceedings{wang2020contournet,
  title={ContourNet: Taking a Further Step toward Accurate Arbitrary-shaped Scene Text Detection},
  author={Wang, Yuxin and Xie, Hongtao and Zha, Zheng-Jun and Xing, Mengting and Fu, Zilong and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11753--11762},
  year={2020}
  }
 ```

## Feedback
Suggestions and discussions are greatly welcome. Please contact the authors by sending email to ```wangyx58@mail.ustc.edu.cn```
