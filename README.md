# RESNET18 ARCHITECTURE FROM SCRATCH
ResNet18 is a Residual Network with 18 layers
Its main idea is:

- Stack residual blocks to make a deep network trainable.

- Use skip connections to avoid vanishing gradients.

- Gradually increase channels and reduce spatial dimensions while preserving information.

### Residual Block solve the issue of vanishing gradient by introducing skip connections
Y=F(X)+X
WHERE:
- X=input
- F(X) → output of some convolutional layers
- Adding X back → ensures the network can learn an identity mapping easily if needed.

## ARCHITECTURE OF RESIDUAL BLOCK
1. Two 3x3 convulationals
   - both have same number of output channels
   - both are followed by BN and ReLU (except the second conv only BN before addition)
2. Skip connection
   - Add input directly before the final ReLU operation
3. Shape matching (1x1 convulational layer introduction)
   - X and F(X) should have the same shape before addition, so that is why we introduce 1x1 convulational channel to change the number of channels to      transform the input into the desired share for the addition operation
  

## RestNet ARCHITECTURE
1. Input Layer
   - Input image shape (224, 224, 3) for RGB images (different for grayscale images like X-Ray)

2. Initial Convolution + BatchNorm + ReLU + MaxPool
   Layers
   - Conv2D (7x7 kernel, 64 filters, strides=2)
      - Large kernel helps capture low-level features (edges, textures) over a wider area.
      - Stride=2 → reduces spatial dimensions from 224x224 → 112x112.
   - BN
   - ReLU
   - MaxPool(3x3, strides=2)
     - Downsamples features → reduces size to 56x56.
     - Helps network to focus on important region, reduces computation

3. Residual Block Groups
   ResNet18 has 4 groups of residual blocks, each group containing 2 residual blocks:

   - Group 1: 64 channels
     - 2 Residual blocks, each with 2 conv layers(3x3), 64 channels
     - Stride=1 → spatial size stays 56x56.
     - Skip connections = identity (no 1x1 conv needed, because shape doesn’t change).
     - Purpose: learn low-level features while preserving size.
   - Group 2: 128 channels
     - 2 Residual blocks, each with 2 conv layers (3x3), 128 channels.
     - First block uses stride=2 + optional 1x1 conv on skip, downsample 56x56 to 28x28.
     - Second block uses stride=1, keeps size 28x28.
     - Purpose: learn higher-level features, increase representational power.
       - Channels increase → more filters capture more complex patterns.
    - Group 3: 256 channels
      - 2 Residual blocks, 256 channels.
      - First block with strides=2, downsample 28x28 to 14x14
      - Second block with stride=1, size remains same
      - Purpose: capture even more abstract features for classification.
    - Group 4: 512 channels
      - 2 Residual blocks, 512 channels.
      - First block: stride=2, downsample 14x14 to 7x7
      - Second block: stride=1 size 7x7
    
    As spatial dimensions shrink, we increase the number of channels to preserve information capacity.

   - Group 1: 64 channels - captures edges & small textures
   - Group 2: 128 channels - more patterns, larger structures
   - Group 3: 256 channels - even more abstract features
   - Group 4: 512 channels - high-level semantic representation

5. Global Average Pooling
   - converts 7x7x512 to 1x1x512
   - This reduces number of parameters drastically compared to flattening → reduces overfitting.

6. FCNN
   - Uses sigmoid for binary or softmax for multiclass
