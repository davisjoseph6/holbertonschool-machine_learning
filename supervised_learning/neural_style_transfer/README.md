# Neural Style Transfer

This project demonstrates the implementation of Neural Style Transfer (NST), a technique that combines the content of one image with the style of another image. By leveraging a pre-trained VGG19 model, this project generates stylized images using content and style cost functions.

---

## Project Overview

The `NST` class performs key operations for neural style transfer, including:
- **Scaling and preprocessing images**
- **Extracting style and content features using a modified VGG19 model**
- **Computing Gram matrices for style representation**
- **Optimizing a generated image using gradient descent**

---

## Files and Directories

### Python Scripts
1. **`0-neural_style.py` to `10-neural_style.py`**
   - Incremental implementation of the `NST` class with functionalities such as:
     - Image scaling
     - Model loading
     - Feature extraction
     - Cost computation (content, style, and total cost)
     - Image generation with gradient descent

2. **`0-main.py` to `10-main.py`**
   - Scripts to test specific features or methods of the `NST` class.

### Supporting Files
- **`golden_gate.jpg`**: Sample content image.
- **`starry_night.jpg`**: Sample style image.
- **`vgg_base.h5`**: Custom VGG19 model with average pooling instead of max pooling for enhanced feature extraction.

---

## Class: `NST`

### Key Methods

1. **Initialization**:
   - Loads content and style images, and initializes the VGG19 model.
   - Parameters include:
     - `alpha`: Weight for content cost.
     - `beta`: Weight for style cost.
     - `var`: Weight for variational loss.

2. **Image Preprocessing**:
   - **`scale_image`**: Rescales images to a maximum dimension of 512 px, normalizes pixel values, and expands dimensions.

3. **Feature Extraction**:
   - **`load_model`**: Loads a modified VGG19 model with selected layers for style and content feature extraction.
   - **`generate_features`**: Extracts and processes Gram matrices for style layers and content features.

4. **Cost Functions**:
   - **`gram_matrix`**: Computes the Gram matrix for a given layer.
   - **`content_cost`**: Measures similarity in content features between the generated and content images.
   - **`style_cost`**: Computes style mismatch across multiple layers using Gram matrices.
   - **`variational_cost`**: Penalizes noise and ensures spatial smoothness in the generated image.

5. **Gradient Descent**:
   - **`compute_grads`**: Computes gradients of the total cost with respect to the generated image.
   - **`generate_image`**: Optimizes the generated image using Adam optimizer for a specified number of iterations.

---

## How to Run

1. **Set Up Environment**:
   - Install TensorFlow, NumPy, and other dependencies:
     ```bash
     pip install tensorflow numpy
     ```

2. **Run a Main Script**:
   - Use `10-main.py` to perform full neural style transfer:
     ```bash
     python3 10-main.py
     ```

3. **Customize Parameters**:
   - Modify `alpha`, `beta`, and `var` in the script to adjust the balance between content, style, and smoothness.

4. **Generated Image**:
   - The stylized image is returned as a NumPy array and can be saved or visualized.

---

## Example Results

### Input
- **Content Image**: `golden_gate.jpg`
- **Style Image**: `starry_night.jpg`

### Output
A stylized image that retains the structure of `golden_gate.jpg` while applying the artistic style of `starry_night.jpg`.

---

## Applications
- Artistic image generation
- Style transfer for video frames
- Creative content creation

---

## References
- *"A Neural Algorithm of Artistic Style"* - Gatys et al. (2015)
- *[TensorFlow Documentation](https://www.tensorflow.org/)*

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davis-joseph/))

