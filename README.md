# DeepSaliency Project

DeepSaliency is a saliency object detection framework based on fully convolutional neural networks with global input (whole raw images) and global output (whole saliency maps). 

###Project Website

[http://www.zhaoliming.net/research/deepsaliency](http://www.zhaoliming.net/research/deepsaliency)

In the project website, you can find detailed descriptions, models, result maps and datasets used in our paper.

Contact: Liming Zhao ([zhaoliming@zju.edu.cn](mailto: zhaoliming@zju.edu.cn))

### Paper

Xi Li, Liming Zhao, Lina Wei , Ming-Hsuan Yang, Fei Wu, Yueting Zhuang, Haibin Ling, Jingdong Wang. "DeepSaliency: Multi-Task Deep Neural Network Model for Salient Object Detection“. IEEE Transactions on Image Processing (TIP), 2016.


### Dependencies

- Caffe (included in the project code)
- Python (ipython notebook is used)
- Linux (Windows is also OK with modification)

### Usage

- Download or clone the project code
- In the `models` directory, download the models from [google drive](https://drive.google.com/folderview?id=0By55MQnF3PHCbFpocU5jOTdVOHM&usp=sharing)
- Then a demo for processing one input image can be found in `demo.ipynb`.


### Training

- Download dataset to `dataset` directory
- Run `dataset\create_caffe_data.py` to obtain the hdf5 training data
- Then training using the script in `models\finetune.sh`
