# Objects as Points
Object detection, 3D detection, and pose estimation using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*         


Contact: [zhouxy@cs.utexas.edu](mailto:zhouxy@cs.utexas.edu). Any questions or discussions are welcomed! 

## Step 1
Prepare your own dataset folder,the structure is same as below:
~~~
--my_dataset
  --images
  --annotations
~~~
images folder contains all the images in the whole dataset you aim to train on 
annotation folder contains the corresponding annotation with json format

## Step 2
Modify the corresponding file:
~~~
1.vim CenterNet/src/lib/datasets/dataset/mydataset.py
~~~
line 14 class MYDATASET(data.Dataset):
line 15 num_classes=*your num class*
line 24 self.data_dir=*your dir*
line 43 self.class_name=*your class name*
line 44 self._valid_ids=*your valid ids*

~~~
2.vim CenterNet/src/lib/datasets/dataset_factory
~~~
add 'mydataset'= MYDATASET to dataset_factory dict

~~~
3.vim CenterNet/src/lib/opts.py
~~~
line 15 default=mydataset

~~~
4.vim CenterNet/src/lib/detectors/ctdet.py
~~~
line 336 num_classes=*your num classes*
line 336 dataset = 'mydataset'

~~~
5.vim CenterNet/src/lib/utils/debugger.py
~~~
line 47 elif num_classes=*your num classes* or dataset == 'mydataset'
line 460 mydataset_class_name = [...]


