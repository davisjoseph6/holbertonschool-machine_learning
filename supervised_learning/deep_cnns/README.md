# Deep Convolutional Neural Networks (Deep CNNs)

This project focuses on implementing and understanding various deep convolutional neural network (CNN) architectures, including Inception, ResNet, and DenseNet. It demonstrates building blocks, such as inception modules, identity blocks, and dense blocks, to create complex and powerful models.

---

## Directory Overview

### Inception Networks
1. **`0-inception_block.py`**
   - Implements an inception block as described in *"Going Deeper with Convolutions"* (2014).

2. **`1-inception_network.py`**
   - Constructs the Inception network using inception blocks for image classification.

---

### Residual Networks (ResNet)
3. **`2-identity_block.py`**
   - Implements an identity block for ResNet, which skips connections to preserve gradient flow.

4. **`3-projection_block.py`**
   - Implements a projection block for ResNet, used for downsampling and increasing the number of filters.

5. **`4-resnet50.py`**
   - Builds the ResNet-50 architecture using identity and projection blocks as described in *"Deep Residual Learning for Image Recognition"* (2015).

---

### Dense Networks (DenseNet)
6. **`5-dense_block.py`**
   - Implements a dense block, where each layer is connected to every other layer as described in *"Densely Connected Convolutional Networks"* (2016).

7. **`6-transition_layer.py`**
   - Implements a transition layer to reduce feature map dimensions using compression in DenseNet.

8. **`7-densenet121.py`**
   - Builds the DenseNet-121 architecture using dense blocks and transition layers.

---

## Key Features

### Architectures
- **Inception**: Employs multi-scale processing with inception blocks for better feature extraction.
- **ResNet**: Introduces residual connections to mitigate vanishing gradients and enable deeper networks.
- **DenseNet**: Improves feature reuse and efficiency through dense connectivity between layers.

### Modular Implementation
- Each building block (e.g., inception block, identity block, dense block) is implemented as a reusable function, enabling flexibility and easy integration.

---

## How to Use

1. **Inception**
   - Use `1-inception_network.py` to create the full Inception network for image classification.

2. **ResNet**
   - Use `4-resnet50.py` to build the ResNet-50 model with pre-defined identity and projection blocks.

3. **DenseNet**
   - Use `7-densenet121.py` to construct the DenseNet-121 model for efficient and high-accuracy classification tasks.

---

## Applications
- Image classification tasks
- Transfer learning with deep CNN architectures
- Exploring advanced deep learning techniques

---

## Requirements
- Python 3.x
- TensorFlow 2.x

---

## References
- *"Going Deeper with Convolutions"* (2014)
- *"Deep Residual Learning for Image Recognition"* (2015)
- *"Densely Connected Convolutional Networks"* (2016)

---

## Author
- Davis Joseph ([LinkedIn]([https://www.linkedin.com/in/davis-joseph/](https://www.linkedin.com/in/davisjoseph767/)))

