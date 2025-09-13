# CelebA Experiment Hyperparameter Summary

## Quick Reference Table

| Model | Beta | Input Dim | Hidden Dim | Latent Dim | N Classes | Learning Rate | Batch Size | Image Size | Epochs | Optimizer | LR Scheduler | Scheduler Patience | Scheduler Factor | Dropout Rate | Grad Clip Norm | N Probes | Enforcement Weight | Probe Dims | Target Distribution |
| ----- | ---- | --------- | ---------- | ---------- | --------- | ------------- | ---------- | ---------- | ------ | --------- | ------------ | ------------------ | ---------------- | ------------ | -------------- | -------- | ------------------ | ---------- | ------------------- |
| Standard_VAE | 1.0 | 12288 | 512 | 64 | 2 | 0.0001 | 32 | 64 | 10 | Adam | ReduceLROnPlateau | 3 | 0.5 | 0.2 | 1.0 | N/A | N/A | N/A | N/A |
| Beta_VAE_0.1 | 0.1 | 12288 | 512 | 64 | 2 | 0.0001 | 32 | 64 | 10 | Adam | ReduceLROnPlateau | 3 | 0.5 | 0.2 | 1.0 | N/A | N/A | N/A | N/A |
| DERP_VAE_5probes | 1.0 | 12288 | 512 | 64 | 2 | 0.0001 | 32 | 64 | 10 | Adam | ReduceLROnPlateau | 3 | 0.5 | 0.2 | 1.0 | 5 | 0.5 | 64 | Normal |

## Detailed Configuration

### Standard_VAE

#### Architecture
- **Model Type**: Standard VAE
- **Input Dim**: 12288
- **Hidden Dim**: 512
- **Latent Dim**: 64
- **N Classes**: 2
- **Encoder Layers**: 12288 -> 512 -> 256 -> 64
- **Decoder Layers**: 64 -> 256 -> 512 -> 12288
- **Classifier Layers**: 64 -> 32 -> 2
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 3
- **Scheduler Factor**: 0.5
- **Batch Size**: 32
- **Epochs**: 10
- **Gradient Clip Norm**: 1.0
- **Beta**: 1.0

#### Dataset
- **Name**: CelebA
- **Image Size**: 64x64
- **Channels**: 3
- **Num Samples**: Full dataset (202,599)
- **Target Attribute**: Smiling
- **Preprocessing**: Resize to 64x64, normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 1.0
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

### Beta_VAE_0.1

#### Architecture
- **Model Type**: Standard VAE
- **Input Dim**: 12288
- **Hidden Dim**: 512
- **Latent Dim**: 64
- **N Classes**: 2
- **Encoder Layers**: 12288 -> 512 -> 256 -> 64
- **Decoder Layers**: 64 -> 256 -> 512 -> 12288
- **Classifier Layers**: 64 -> 32 -> 2
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 3
- **Scheduler Factor**: 0.5
- **Batch Size**: 32
- **Epochs**: 10
- **Gradient Clip Norm**: 1.0
- **Beta**: 0.1

#### Dataset
- **Name**: CelebA
- **Image Size**: 64x64
- **Channels**: 3
- **Num Samples**: Full dataset (202,599)
- **Target Attribute**: Smiling
- **Preprocessing**: Resize to 64x64, normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 0.1
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

### DERP_VAE_5probes

#### Architecture
- **Model Type**: DERP-VAE
- **Input Dim**: 12288
- **Hidden Dim**: 512
- **Latent Dim**: 64
- **N Classes**: 2
- **Encoder Layers**: 12288 -> 512 -> 256 -> 64
- **Decoder Layers**: 64 -> 256 -> 512 -> 12288
- **Classifier Layers**: 64 -> 32 -> 2
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 3
- **Scheduler Factor**: 0.5
- **Batch Size**: 32
- **Epochs**: 10
- **Gradient Clip Norm**: 1.0
- **Beta**: 1.0

#### Dataset
- **Name**: CelebA
- **Image Size**: 64x64
- **Channels**: 3
- **Num Samples**: Full dataset (202,599)
- **Target Attribute**: Smiling
- **Preprocessing**: Resize to 64x64, normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 1.0
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

#### Derp Specific
- **N Probes**: 5
- **Enforcement Weight**: 0.5
- **Probe Dimensions**: 64
- **Target Distribution**: Standard Normal
- **Ks Distance Type**: Modified (average deviation)
- **Perceptual Loss Weight**: 0.01
- **Perceptual Loss Layers**: ['relu1_2', 'relu2_2', 'relu3_3']

