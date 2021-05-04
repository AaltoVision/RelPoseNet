# RelPoseNet
A PyTorch version of the ego-motion estimation pipeline proposed in [our work](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf). The official implementation (in Lua) is available at https://github.com/AaltoVision/camera-relocalisation

## Evaluation on the [7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
scene|[Lua](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf)| PyTorch (this repo)
:---|:---:|:---:
Chess|0.13m, 6.46deg|0.12m, 7.10deg
Fire |0.26m, 12.72deg|0.26m, 12.45deg
Heads|0.14m, 12.34deg|0.14m, 11.72deg
Office|0.21m, 7.35deg|0.20m, 9.23deg
Pumpkin|0.24m, 6.35deg|0.21m, 8.10deg
Red Kitchen|0.24m, 8.03deg|0.23m, 8.82deg
Stairs|0.27m, 11.82deg|0.27m, 11.66deg
Average|0.21m, 9.30deg|0.20m, 9.87deg
<<<<<<< HEAD

## Installation
- create and activate conda environment with Python 3.x
```
conda create -n my_fancy_env python=3.7
source activate my_fancy_env
```
- install all dependencies by running the following command:
```
pip install -r requirements.txt
```

## Evaluation and Training
Evaluation and training have been performed on the 7-Scenes dataset available [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/). Important!!! The images have to be resized such that the smaller dimension is 256 and the aspect ratio is intact. This could be done using the following command:
```find . -name "*.color.png" | xargs -I {} convert {} -resize "256^>" {}```

### Evaluation

### Training

https://drive.google.com/drive/folders/1TnVuR2bNZviYYdT3XLqCW4xjO19eLG6T?usp=sharing




## License
Our code is released under the Creative Commons BY-NC-SA 3.0, available only for non-commercial use.

## How to cite
If you use this project in your research, please cite:

```
@inproceedings{Laskar2017PoseNet,
      title = {Camera relocalization by computing pairwise relative poses using convolutional neural network},
      author = {Laskar, Zakaria and Melekhov, Iaroslav and Kalia, Surya and Kannala, Juho},
       year = {2017},
       booktitle = {Proceedings of the IEEE International Conference on Computer Vision Workshops}
}

@inproceedings{Melekhov2017RelPoseNet,
      title = {Camera relocalization by computing pairwise relative poses using convolutional neural network},
      author = {Melekhov, Iaroslav and Ylioinas, Juha and Kannala, Juho and Rahtu, Esa},
       year = {2017},
       booktitle = {International Conference on Advanced Concepts for Intelligent Vision Systems}
}
```

=======
>>>>>>> 44ba94e53ebec7873eddd0558065240c272c1b9e
