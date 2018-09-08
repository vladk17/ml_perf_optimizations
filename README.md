
# Notes on Performance Optimizations in ML

PURPOSE: PRIVATE WORKING NOTES<br>
STATUS:  VERY INITIAL DRAFT

## Abstract:
TBD

### TOC
* [1. Introduction](#section1)
* [2. CNNs](#section2)
    * [2.1 Champion CNN Architectures for Image Classification Task](#section2.1)
        * [2.1.1 Model Compressions](#section2.1.1)
        * [2.1.2 CNN Micro Architecture](#section2.1.2)
        * [2.1.3 CNN Macro Architecture](#section2.1.3)
        * [2.1.4 Neural Network Design Space Exploration](#section2.1.4)
* [3. Small CNNs](#section3)
    * [3.1 Squeeznet](#section3.1)
    * [3.2 SqueezeNext](#section3.2)
    * [3.4 MobileNets](#section3.3)
    * [3.4 See Also](#section3.4)
* [4. Other relevant topics](#section4)
    * [4.1. Search for Minimal Representations](#section4.1)
    * [4.2. Reverse Engineering of model decision passes](#section4.2)
        * [4.2.1. Visualizations what CNNs learn](#section4.2.1)
    * [4.3. "Introduction to deep learning efficiency" by Amir Alush, brodmann17](https://youtu.be/x5C9XnYanLw)
    * [4.4. "Tips and Tricks for Developing Smaller Neural Nets" by Forrest Iandola, CEO of DeepScale.](https://www.youtube.com/watch?v=N-HnlYlhb18)
    * [4.5. Second presentation from brodmann17, where different models and image processing tasks are presented - compare the models and their sizes](#section4.5)
    * [4.6. Reduce Precision (quantization) of Weights and Activations](#section4.6)
    * [4.7. ONNX](#section4.7)
    * [4.8. TensorRT](#section4.8)    
    * [4.9 HW/SW co-design](#section4.9)
    * [4.10. Symmetries](#section4.10)    
    * [4.11. All optical NNs](#section4.11)
    * [4.12. NVIDIA's TensorRT](#section4.12)
    * [NVIDIA Deep Learning Accelerator (NVDLA)](http://nvdla.org/)

<a id='section1'></a>
## 1. Introduction

Here I am planning to collect notes on performance optimizations of machine learning models<br>

I will start with CNNs, but am planning to consider also RNNs/LSTMs on one side, and "classical" models and approaches, like RandomForest, XGBoost, etc. on the other side.  

Besides _accuracy_, the performance characteristics of machine learning models include the _model size_, _power consumption_, _speed_ of learning (forward and backward time) and inference, etc... TBD<br>

Most typical performance limiter is DRAM access bound. Therefore, the general optimization approach is to find a way to reduce the number of DRAM accesses (do more things within registers and within internal memory of a chip).  

Small model size is a key to improving of other performance characteristics of a model [1]:
1. Smaller CNNs require less communication across servers during distributed training.  
2. Smaller CNNs require less bandwidth to export a new model from the
cloud to an autonomous car. 
3. Smaller CNNs are more feasible to deploy on FPGAs and other hardware with limited internal memory. (Fitting the model to internal memory saves tons of energy. Internal memory accesses are few orders of magnitude faster=>faster inference)

The question that we are going to address is __what is the minimal model for a given accuracy level__.

<a id='section2'></a>
## 2. CNNs

Computer Vision Tasks:

* Classification
* Classification + Localization
* Object Detection
* Image Segmentation

<a id='section2.1'></a>
### 2.1 Champion CNN Architectures for Image Classification Task


 <table style="width:100%">
  <tr>
    <th>name</th>
    <th>year, author</th>
    <th>paper</th>
  </tr>
  <tr>
    <td>LeNet5</td>
    <td>(1989, LeCun)</td>
    <td>
        <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">"Gradient-based learning applied to document recognition"</a>
    </td>
  </tr>
  <tr>
    <td>AlexNet</td>
    <td>(2012, krizhevsky)</td>
    <td>
        <a href = "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">"ImageNet Classification with Deep Convolutional Neural Networks"</a>
    </td>
  </tr>
  <tr>
    <td>VGG16/19</td>
    <td>(2014, Symonyan)</td>
    <td>
        <a href="https://arxiv.org/abs/1409.1556">"Very Deep Convolutional Networks for Large-Scale Image Recognition"</a> 
     </td>
  </tr>
  <tr>
    <td>GoogLeNet</td>
    <td>(2014, Szegedy)</td>
    <td><a href="https://arxiv.org/abs/1409.4842">"Going deeper with convolutions"</a>
    </td>
  </tr>
  <tr>
    <td>Inception V1-V3</td>
    <td>(2015, Szegedy)</td>
    <td>
        <a href="https://arxiv.org/abs/1502.03167">"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"</a><br>
        <a href="https://arxiv.org/abs/1512.00567">"Rethinking the Inception Architecture for Computer Vision"</a>
     </td>
  </tr>
  <tr>
    <td>Residual Networks</td>
    <td>(2016, He)</td>
    <td>
        <a href="https://arxiv.org/abs/1512.03385">"Deep Residual Learning for Image Recognition"</a> 
    </td>
  </tr>
  <tr>
    <td>DenseNet</td>
    <td>(2017, Huang)</td>
    <td>
        <a href="https://arxiv.org/abs/1608.06993">"Densely Connected Convolutional Networks"</a>
    </td>
  </tr>
  <tr>
    <td>ResNeXT</td>
    <td>(2017, Xie)</td>
    <td>
        <a href="https://arxiv.org/abs/1611.05431">"Aggregated Residual Transformations for Deep Neural Networks"</a>
    </td>
  </tr>
    
</table> 

<br>


["CNN Architectures" secture from cs231n](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf)

["Efficient Methods and Hardware for Deep Learning" from cs231n](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf)

["An Analysis of Deep Neural Network Models for Practical Applications" by A. Canziani, A. Paszke, E. Culurciello](https://arxiv.org/pdf/1605.07678.pdf)


["Fast Algorithms for Convolutional Neural Networks" by Andrew Lavin, Scott Gray](https://arxiv.org/pdf/1509.09308.pdf)

["Distilling the Knowledge in a Neural Network" by Geoffrey Hinton, Oriol Vinyals, Jeff Dean](https://arxiv.org/abs/1503.02531)

>Note: For memory bound workloads CPU may give beter performance than GPU 

<a id='section2.1.1'></a>
#### 2.1.1 Model Compressions
TBD
<a id='section2.1.2'></a>
#### 2.1.2 CNN Micro-Architecture
TBD<br>
"We use the term CNN Micro-Architecture to refer to the particular organization and dimensions of the individual modules"

<a id='section2.1.3'></a>
#### 2.1.3 CNN Macro-Architecture
TBD<br>
"While  the  CNN  Micro-Architecture  refers  to  individual  layers  and  modules,  we  define  the CNN Macro-Architecture as the system-level organization of multiple modules into an end-to-end CNN architecture."<br>
Study impact of depth on occuracy

<a id='section2.1.4'></a>
#### 2.1.4 Neural Network Design Space Exploration
"developing automated approaches for finding NN architectures that deliver higher accuracy"

<a id='section3'></a>
## 3. Small CNNs

<a id='section3.1'></a>
### 3.1 Squeeznet
PRESERVING ACCURACY  WITH  FEW PARAMETERS<br>
https://github.com/DeepScale/SqueezeNet <br>
SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.  Additionally, with model compression techniques, we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet)

<a id='section3.2'></a>
### 3.2 SqueezeNext
"If you like SqueezeNet, you might also like SqueezeNext!" ([SqueezeNext paper](https://arxiv.org/abs/1803.10615), [SqueezeNext code](https://github.com/amirgholami/SqueezeNext))

<a id='section3.3'></a>
### 3.3 MobileNets
[MobileNets](https://arxiv.org/abs/1704.04861)


<a id='section3.4'></a>
### 3.4 See Also

See also [Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device](https://arxiv.org/abs/1806.05363)

See also [What is the Kirin 970's NPU? ](https://www.youtube.com/watch?v=A6ouKQjvSmw)

See also ["Optimal Brain Damage", LeCun et al. 1990](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf)

See also ["Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Prunning", Yang et al. 2017](https://arxiv.org/abs/1611.05128)

See also ["Learning bith weights and connections for efficient neural networks", Yang et al 2015](https://arxiv.org/abs/1506.02626)

See also ["EIE efficient inference engine on compressed deep neural network", Han et. al 2016](https://arxiv.org/abs/1602.01528)


Look also at [DawnBench - An End-to-End Deep Learning Bemnchmark and Competition](https://dawn.cs.stanford.edu/benchmark/)

<a id='section4'></a>
## 4. Other relevant topics

<a id='section4.1'></a>
### 4.1. Search for Minimal Representations
[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538?context=cs)

[Deep Learning of Representations: Looking Forward](https://arxiv.org/abs/1305.0445)

<a id='section4.2'></a>
### 4.2. Reverse Engineering of model decision passes

<a id='section4.2.1'></a>
#### 4.2.1. Visualizations what CNNs learn
* Visualizing intermediate activations
* Visualizing filters 
* Visualizing heatmaps of activations in an image

<a id='section4.6'></a>
### 4.6. Reduce Precision (quantization) of Weights and Activations

[Mixed Precision Training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)

<a id='section4.7'></a>
### 4.7 ONNX
Standard for sharing of DNN models<br>
https://github.com/onnx/onnx <br>
Export your model to ONNX and the use it for inference on an other platform including mobile etc.

pytorch -> ONNX -> caffe2

## References

[1] F. Iandola, S. Han, M. Moskewicz, Kh. Ashraf, W. Dally, Kurt KeutzerSQUEEZENET:ALEXNET-LEVEL ACCURACY   WITH50X FEWER PARAMETERS AND<0.5MB MODEL SIZE (https://arxiv.org/pdf/1602.07360.pdf)<br>
@article{SqueezeNet,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}
