
# Performance Optimizations in ML

## Abstract:
TBD

### TOC
* [0. structured data models (Random Forest, XGBoost) optimizations](#section0)
* [1. Squeeznet](#section1)
* [1. LSTM?](#section1)
* [2. Search for Minimal Representations](#section2)
* [3. Reverse Engineering of model decision passes](#section3)
* [4. Introduction to deep learning efficiency by Amir Alush](https://youtu.be/x5C9XnYanLw)
* [4 Forrest Iandola, CEO of DeepScale. Tips and Tricks for Developing Smaller Neural Nets](https://www.youtube.com/watch?v=N-HnlYlhb18)
* [5. Second presentation from brodman17, where different models and image processing tasks are presented - compare the models and their sizes](#section5)
* [6. Reduce Precision (quantization) of Weights and Activations](#section6)
* [7. ONNX](https://github.com/onnx/onnx)
* [8 HW/SW co-design](#section8)
* [9. Symmetries](#section9)
* [10. Visualizations are the must!](#secrion10)
* [11. All optical NNs](#section11)
* [12. Computer Vision Tasks](#section12)
    * [12.1 Classification](#section12.1)
    * [12.2 Classification + Localization](#section12.2)
    * [12.3 Object Detection](#section12.3)
    * [12.4 Image Segmentation](#section12.4)
* [NVIDIA's TensorRT](#section13)

Here I am going to focus on performance optimizations of machine learning models<br>
In addition to occuracy of the model, we are considering the model size, power consumption, etc... TBD<br>

The question that we are going to concider is what is the minimal model for a given occuracy level. 

Advantages of small models according to [1] are:
1. Smaller CNNs require less communication across servers during distributed training.  
2. Smaller CNNs require less bandwidth to export a new model from the
cloud to an autonomous car. 
3. Smaller CNNs are more feasible to deploy on FPGAs and other hardware with limited internal memory. (Fitting the model to internal memory saves tons of energy. Internal memory accesses are few orders of magnitude faster=>faster inference)

## Squeeznet
https://github.com/DeepScale/SqueezeNet <br>
SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.  Additionally, with model compression techniques, we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet)

If you like SqueezeNet, you might also like SqueezeNext! ([SqueezeNext paper](https://arxiv.org/abs/1803.10615), [SqueezeNext code](https://github.com/amirgholami/SqueezeNext))

See also [MobileNets](https://arxiv.org/abs/1704.04861)

See also [Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device](https://arxiv.org/abs/1806.05363)

See also [What is the Kirin 970's NPU? ](https://www.youtube.com/watch?v=A6ouKQjvSmw)

See also ["Optimal Brain Damage", LerCun et al. 1990](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf)

See also ["Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Prunning", Yang et al. 2017](https://arxiv.org/abs/1611.05128)

See also ["Learning bith weights and connections for efficient neural networks", Yang et al 2015](https://arxiv.org/abs/1506.02626)

See also ["EIE efficient inference engine on compressed deep neural network", Han et. al 2016](https://arxiv.org/abs/1602.01528)


Look also at [DawnBench - An End-to-End Deep Learning Bemnchmark and Competition](https://dawn.cs.stanford.edu/benchmark/)

### Related Work

#### x.0 Evolution of CNN Architectures for Classification

|name             |                  |        
|---------------- | -----------------| ------- 
|LeNet5           |(1989, LeCun)     | ["Gradient-based learning applied to document recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
|AlexNet          |(2012, krizhevsky)|["ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 
|VGG16/19         |(2014, Symonyan)  |["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) 
|GoogLeNet        |(2014, Szegedy)   |["Going deeper with convolutions"](https://arxiv.org/abs/1409.4842) 
|Inception V1-V3  |(2015, Szegedy)   |["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167)<br>, ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567)
|Residual Networks|(2016, He)        |["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) 
|DenseNet         |(2017, Huang)     |["Densely Connected Convolutional Networks"](https://arxiv.org/abs/1608.06993)
|ResNeXT          |(2017, Xie)       |["Aggregated Residual Transformations for Deep Neural Networks"](https://arxiv.org/abs/1611.05431)|

...<br>
...
 <table style="width:100%">
  <tr>
    <th>Firstname</th>
    <th>Lastname</th>
    <th>Age</th>
  </tr>
  <tr>
    <td>Jill</td>
    <td>Smith</td>
    <td>50</td>
  </tr>
  <tr>
    <td>Eve</td>
    <td>Jackson</td>
    <td>94</td>
  </tr>
</table> 

...<br>
...



#### x.1 model Compression
TBD
#### x.2 CNN Micro Architecture
TBD<br>
"We use the term CNN microarchitecture to refer to the particular organization and dimensions of the individual modules"
#### x.3 CNN Macro Architecture
TBD<br>
"While  the  CNN  microarchitecture  refers  to  individual  layers  and  modules,  we  define  the CNN macroarchitecture as the system-level organization of multiple modules into an end-to-end CNN architecture."<br>
Study impact of depth on occuracy
#### x.4 Neural Network Design Space Exploration
"developing automated approaches for finding NN architectures that deliver higher accuracy"

### 3. SQUEEZE NET: PRESERVING ACCURACY  WITH  FEW PARAMETERS

### Search for Minimal Representations
[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538?context=cs)

[Deep Learning of Representations: Looking Forward](https://arxiv.org/abs/1305.0445)

<a id='section6'></a>
## 6. Reduce Precision (quantization) of Weights and Activations



<a id='section12'></a>
## 12. Computer Vision Tasks

* Classification
* Classification + Localization
* Object Detection
* Image Segmentation

## References

[1] F. Iandola, S. Han, M. Moskewicz, Kh. Ashraf, W. Dally, Kurt KeutzerSQUEEZENET:ALEXNET-LEVEL ACCURACY   WITH50X FEWER PARAMETERS AND<0.5MB MODEL SIZE (https://arxiv.org/pdf/1602.07360.pdf)<br>
@article{SqueezeNet,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}
