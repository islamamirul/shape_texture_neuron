#  Shape or Texture: Understanding Discriminative Features in CNNs, ICLR 2021

**[Shape or Texture: Understanding Discriminative Features in CNNs](https://openreview.net/forum?id=NcFEZOi-rLa)**
<br>
**[Md Amirul Islam](https://www.cs.ryerson.ca/~amirul/)**, **[Matthew Kowal](https://mkowal2.github.io/)**, **[Patrick Esser](https://github.com/pesser/)**, **[Sen Jia](https://scholar.google.com/citations?user=WOsy1foAAAAJ&hl=en)**, **[Björn Ommer](https://hci.iwr.uni-heidelberg.de/people/bommer)**, **[Konstantinos G. Derpanis](https://www.cs.ryerson.ca/~kosta/)**, **[Neil Bruce](http://socs.uoguelph.ca/~brucen/)** 

<br>

![Alt text](dim_estimation/assets/dim_est.png?raw=true "Title")
<br>

#  Data Preparation
**Stylized PASCAL VOC 2012 Preparation**

1. Download the PASCAL VOC 2012 dataset from [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

2. Run the following script to generate the stylized voc 2012 dataset

            cd generate_stylized_voc
            python make_stl_voc.py


#  Estimating shape and texture neurons

1. Change --data_path in config to point to the stylized VOC dataset location

2. To estimate the shape and texture neurons in a ResNet50 and ViT-16-Base-224 using SVOC (args and results saved to --save_dir), run:

            python main.py --model resnet50 --image_size 513 --save_dir dim_outputs/svoc/resnet50
            python main.py --model vit_base_patch16_224  --image_size 224 --save_dir dim_outputs/svoc/ViT-16-Base-224

Note that the results are in the format \[SHAPE, TEXTURE, RESIDUAL\] and display both the number of neurons and percentages. 

For more examples of run configurations, see the files 'run_CNNs_dim_estimation.sh' and 'run_transformer_dim_estimation.sh'

If you want to use your own model, you need it to output a global-pooled latent representation. For an example, see the '_forward_impl' function in models.resnet.py. The models in the 'models' directory are changed in this way already. You should also load the checkpoint in utils.py when the model is selected.

# BibTeX
If you find this repository useful, please consider giving a star :star: and citation :t-rex:


      @InProceedings{islam2021shape,
       title={Shape or texture: Understanding discriminative features in CNNs},
       author={Islam, Md Amirul and Kowal, Matthew and Esser, Patrick and Jia, Sen and Ommer, Bjorn and Derpanis, Konstantinos G and Bruce, Neil},
       booktitle={International Conference on Learning Representations},
       year={2021}
     }


