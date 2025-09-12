# CIFAR-10 Experiment Hyperparameter Summary

## Quick Reference Table

| Model | Beta | Input Dim | Hidden Dim | Latent Dim | N Classes | Learning Rate | Batch Size | Epochs | Optimizer | LR Scheduler | Scheduler Patience | Scheduler Factor | Dropout Rate | Grad Clip Norm | N Probes | Enforcement Weight | Probe Dims | Target Distribution |
| ----- | ---- | --------- | ---------- | ---------- | --------- | ------------- | ---------- | ------ | --------- | ------------ | ------------------ | ---------------- | ------------ | -------------- | -------- | ------------------ | ---------- | ------------------- |
| Enhanced_Standard_VAE | 1.0 | 3072 | 256 | 4 | 10 | 0.001 | 128 | 20 | Adam | ReduceLROnPlateau | 5 | 0.5 | 0.2 | 1.0 | N/A | N/A | N/A | N/A |
| Enhanced_Beta_VAE_0.5 | 0.5 | 3072 | 256 | 4 | 10 | 0.001 | 128 | 20 | Adam | ReduceLROnPlateau | 5 | 0.5 | 0.2 | 1.0 | N/A | N/A | N/A | N/A |
| Enhanced_Beta_VAE_2.0 | 2.0 | 3072 | 256 | 4 | 10 | 0.001 | 128 | 20 | Adam | ReduceLROnPlateau | 5 | 0.5 | 0.2 | 1.0 | N/A | N/A | N/A | N/A |
| DERP_VAE_3probes | 1.0 | 3072 | 256 | 4 | 10 | 0.001 | 128 | 20 | Adam | ReduceLROnPlateau | 5 | 0.5 | 0.2 | 1.0 | 3 | 1.0 | 4 | Normal |
| DERP_VAE_5probes | 1.0 | 3072 | 256 | 4 | 10 | 0.001 | 128 | 20 | Adam | ReduceLROnPlateau | 5 | 0.5 | 0.2 | 1.0 | 5 | 1.0 | 4 | Normal |

## Detailed Configuration

### Enhanced_Standard_VAE

#### Architecture
- **Model Type**: Standard VAE
- **Input Dim**: 3072
- **Hidden Dim**: 256
- **Latent Dim**: 4
- **N Classes**: 10
- **Encoder Layers**: 3072 -> 256 -> 128 -> 4
- **Decoder Layers**: 4 -> 128 -> 256 -> 3072
- **Classifier Layers**: 4 -> 2 -> 10
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 5
- **Scheduler Factor**: 0.5
- **Batch Size**: 128
- **Epochs**: 20
- **Gradient Clip Norm**: 1.0
- **Beta**: 1.0

#### Dataset
- **Name**: CIFAR-10
- **Image Size**: 32x32
- **Channels**: 3
- **Num Samples**: 50,000 train / 10,000 test
- **Num Classes**: 10
- **Preprocessing**: Normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 1.0
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

### Enhanced_Beta_VAE_0.5

#### Architecture
- **Model Type**: Standard VAE
- **Input Dim**: 3072
- **Hidden Dim**: 256
- **Latent Dim**: 4
- **N Classes**: 10
- **Encoder Layers**: 3072 -> 256 -> 128 -> 4
- **Decoder Layers**: 4 -> 128 -> 256 -> 3072
- **Classifier Layers**: 4 -> 2 -> 10
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 5
- **Scheduler Factor**: 0.5
- **Batch Size**: 128
- **Epochs**: 20
- **Gradient Clip Norm**: 1.0
- **Beta**: 0.5

#### Dataset
- **Name**: CIFAR-10
- **Image Size**: 32x32
- **Channels**: 3
- **Num Samples**: 50,000 train / 10,000 test
- **Num Classes**: 10
- **Preprocessing**: Normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 0.5
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

### Enhanced_Beta_VAE_2.0

#### Architecture
- **Model Type**: Standard VAE
- **Input Dim**: 3072
- **Hidden Dim**: 256
- **Latent Dim**: 4
- **N Classes**: 10
- **Encoder Layers**: 3072 -> 256 -> 128 -> 4
- **Decoder Layers**: 4 -> 128 -> 256 -> 3072
- **Classifier Layers**: 4 -> 2 -> 10
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 5
- **Scheduler Factor**: 0.5
- **Batch Size**: 128
- **Epochs**: 20
- **Gradient Clip Norm**: 1.0
- **Beta**: 2.0

#### Dataset
- **Name**: CIFAR-10
- **Image Size**: 32x32
- **Channels**: 3
- **Num Samples**: 50,000 train / 10,000 test
- **Num Classes**: 10
- **Preprocessing**: Normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 2.0
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

### DERP_VAE_3probes

#### Architecture
- **Model Type**: DERP-VAE
- **Input Dim**: 3072
- **Hidden Dim**: 256
- **Latent Dim**: 4
- **N Classes**: 10
- **Encoder Layers**: 3072 -> 256 -> 128 -> 4
- **Decoder Layers**: 4 -> 128 -> 256 -> 3072
- **Classifier Layers**: 4 -> 2 -> 10
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 5
- **Scheduler Factor**: 0.5
- **Batch Size**: 128
- **Epochs**: 20
- **Gradient Clip Norm**: 1.0
- **Beta**: 1.0

#### Dataset
- **Name**: CIFAR-10
- **Image Size**: 32x32
- **Channels**: 3
- **Num Samples**: 50,000 train / 10,000 test
- **Num Classes**: 10
- **Preprocessing**: Normalize to [0,1]

#### Loss Components
- **Reconstruction Loss**: BCE
- **Kl Divergence Weight**: 1.0
- **Classification Loss**: CrossEntropy
- **Classification Weight**: 0.1

#### Compute
- **Device**: cpu
- **Precision**: float32

#### Derp Specific
- **N Probes**: 3
- **Enforcement Weight**: 1.0
- **Probe Dimensions**: 4
- **Target Distribution**: Standard Normal
- **Ks Distance Type**: Modified (average deviation)
- **Perceptual Loss Weight**: 0.01
- **Perceptual Loss Layers**: ['relu1_2', 'relu2_2', 'relu3_3']

### DERP_VAE_5probes

#### Architecture
- **Model Type**: DERP-VAE
- **Input Dim**: 3072
- **Hidden Dim**: 256
- **Latent Dim**: 4
- **N Classes**: 10
- **Encoder Layers**: 3072 -> 256 -> 128 -> 4
- **Decoder Layers**: 4 -> 128 -> 256 -> 3072
- **Classifier Layers**: 4 -> 2 -> 10
- **Activation**: ReLU
- **Output Activation**: Sigmoid (reconstruction), None (classification)
- **Dropout Rate**: 0.2

#### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Lr Scheduler**: ReduceLROnPlateau
- **Scheduler Patience**: 5
- **Scheduler Factor**: 0.5
- **Batch Size**: 128
- **Epochs**: 20
- **Gradient Clip Norm**: 1.0
- **Beta**: 1.0

#### Dataset
- **Name**: CIFAR-10
- **Image Size**: 32x32
- **Channels**: 3
- **Num Samples**: 50,000 train / 10,000 test
- **Num Classes**: 10
- **Preprocessing**: Normalize to [0,1]

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
- **Enforcement Weight**: 1.0
- **Probe Dimensions**: 4
- **Target Distribution**: Standard Normal
- **Ks Distance Type**: Modified (average deviation)
- **Perceptual Loss Weight**: 0.01
- **Perceptual Loss Layers**: ['relu1_2', 'relu2_2', 'relu3_3']

