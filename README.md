# Computer Vision Project

A comprehensive machine learning project exploring various computer vision techniques including CNNs, dimensionality reduction, clustering algorithms, language models with RAG, and CLIP model evaluation.

## Project Overview

This repository contains five major components:

1. **Car vs Duck Classification using CNN**
2. **Image Clustering Analysis with Multiple Dimensionality Reduction Techniques**
3. **Conversational Language Models with RAG and Fine-tuning**
4. **CLIP Model Architecture Analysis and Evaluation**
5. **Basic Image Clustering Implementation**

## Repository Structure

```
Computer Vision/
├── Car_Duck_Classification_CNN.ipynb
├── Clustering.ipynb
├── Image_Clustering_Analysis.ipynb
├── Language_Models_RAG_FineTuning.ipynb
├── CLIP_Model_Evaluation.ipynb
├── README.md
└── Data/
    ├── NeuralNetwork/
    └── Clustering/
```

## Files and Contents

### 1. Car_Duck_Classification_CNN.ipynb

**Purpose:** Implements a Convolutional Neural Network for binary image classification between cars and ducks.

**Key Features:**
- Custom CNN architecture with 3 convolutional layers
- Data augmentation with random horizontal flips, rotation, and color jitter
- Training on 140 images, validation on 35 images
- Dropout regularization (30%) to prevent overfitting
- Testing across 5 different image conditions: realistic, features, blurred, geons, and silhouettes

**Architecture:**
- **Input:** RGB images (224×224×3)
- **Conv1:** 32 filters, 3×3 kernel, ReLU activation, MaxPool
- **Conv2:** 64 filters, 3×3 kernel, ReLU activation, MaxPool
- **Conv3:** 128 filters, 3×3 kernel, ReLU activation, MaxPool
- **FC1:** 100,352 → 256 units with dropout
- **FC2:** 256 → 64 units with dropout
- **FC3:** 64 → 2 units (output layer)
- **Total Parameters:** 25,800,194 (25.8M parameters)

**Training Configuration:**
- Optimizer: Adam with learning rate 0.001
- Loss function: Cross-Entropy Loss
- Epochs: 10
- Batch size: 32
- Device: MPS (Metal Performance Shaders) for Apple Silicon

**Results:**
- Final training accuracy: 80.00%
- Final validation accuracy: 74.29%
- Training time: 13.93 seconds
- Average test accuracy across all conditions: 64.89%

**Condition-Specific Test Accuracy:**
- Realistic: 50.00%
- Features: 80.00%
- Blurred: 77.78%
- Geons: 66.67%
- Silhouettes: 50.00%

**Visualizations:**
- Training and validation loss curves
- Training and validation accuracy curves

---

### 2. Clustering.ipynb

**Purpose:** Basic implementation of image clustering using PCA and visualization techniques.

**Key Features:**
- Principal Component Analysis (PCA) for dimensionality reduction
- Data preprocessing with StandardScaler
- Analysis across 5 image conditions: realistic, features, blurred, geons, silhouettes
- Visualization of dimensionality reduction results

**Image Processing:**
- Image size: 224×224 pixels
- Conversion to grayscale
- Flattened to 50,176-dimensional vectors
- Standardization using StandardScaler

**PCA Analysis:**
- Minimum components for 95% variance preservation:
  - Realistic: 30 components (95.31% variance)
  - Features: 33 components (95.73% variance)
  - Blurred: 23 components (95.30% variance)
  - Geons: 39 components (95.20% variance)
  - Silhouettes: 31 components (95.66% variance)

**Object Categories:** Airplane, car, chair, cup, dog, donkey, duck, hat

**Dimensionality Reduction Techniques:**
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- LLE (Locally Linear Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

**Visualizations:**
- 2D scatter plots for each dimensionality reduction technique
- Color-coded by object cluster
- Separate plots for each image condition

---

### 3. Image_Clustering_Analysis.ipynb

**Purpose:** Comprehensive image clustering analysis using K-Means and EM (Gaussian Mixture Models) with multiple feature extraction methods.

**Key Features:**
- Multiple dimensionality reduction techniques (PCA, t-SNE, LLE, UMAP)
- K-Means clustering with optimal k selection
- EM clustering (Gaussian Mixture Models)
- ResNet-18 feature extraction for deep learning-based clustering
- Confusion matrix analysis for clustering accuracy
- Synthetic image generation using GMM

**Dimensionality Reduction:**

**PCA (Principal Component Analysis):**
- Reduces 50,176-dimensional image vectors to components preserving 95% variance
- Visualization of original vs reconstructed images
- 2D scatter plots with explained variance ratios

**t-SNE:**
- Non-linear dimensionality reduction
- Parameters: perplexity=30, random_state=42
- Captures local structure and cluster separation

**LLE (Locally Linear Embedding):**
- Manifold learning technique
- Parameters: n_neighbors=10, random_state=42
- Preserves local neighborhood relationships

**UMAP:**
- Modern manifold learning technique
- Parameters: n_neighbors=15, min_dist=0.1, random_state=42
- Balances local and global structure preservation

**K-Means Clustering:**

**Optimal k Selection:**
- Elbow method using inertia
- Silhouette score analysis
- k range: 2-12 clusters

**Clustering with k=8 (Ground Truth):**
- Uses PCA-reduced features (95% variance)
- Cluster-to-class mapping based on majority voting
- Confusion matrix visualization

**Accuracy by Condition (PCA Features):**
- Realistic: Varies by implementation
- Features: Varies by implementation
- Blurred: Varies by implementation
- Geons: Varies by implementation
- Silhouettes: Varies by implementation

**K-Means with ResNet-18 Features:**
- Pre-trained ResNet-18 model (ImageNet weights)
- Feature extraction from penultimate layer (512-dimensional)
- Images resized to 224×224
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Significantly improved clustering accuracy compared to PCA

**EM Clustering (Gaussian Mixture Models):**

**Model Selection:**
- BIC (Bayesian Information Criterion) for optimal k
- AIC (Akaike Information Criterion) for optimal k
- Covariance type: full
- Random state: 42
- Number of initializations: 10

**Clustering with k=8 (Ground Truth):**
- Uses PCA-reduced features
- Soft clustering with probability assignments
- Confusion matrix visualization with green colormap

**GMM Image Generation:**
- Sampling from learned Gaussian mixtures
- Inverse PCA transformation to reconstruct images
- Generated 5 samples per condition
- Visualization of synthetic images

**EM Clustering with ResNet Features:**
- Applied GMM on deep features from ResNet-18
- Comparison with PCA-based clustering
- Confusion matrix visualization with purple colormap

**ResNet-18 Feature Extraction:**
- Model: torchvision.models.resnet18
- Weights: IMAGENET1K_V1
- Architecture modification: Removed final classification layer
- Output: 512-dimensional feature vectors
- Image preprocessing: Resize to 224×224, normalize with ImageNet stats

**Device Support:**
- MPS (Metal Performance Shaders) for Apple Silicon
- CUDA for NVIDIA GPUs
- CPU fallback

**Visualizations:**
- Elbow curves for K-Means (inertia and silhouette scores)
- BIC and AIC curves for GMM
- Confusion matrices for all clustering methods
- 2D scatter plots for all dimensionality reduction techniques
- PCA reconstruction visualization
- Generated images from GMM

---

### 4. Language_Models_RAG_FineTuning.ipynb

**Purpose:** Implements Retrieval Augmented Generation (RAG), fine-tuning with LoRA, and Model Context Protocol (MCP) server for conversational AI systems.

**Part A: Retrieval Augmented Generation (RAG)**

**Model:** microsoft/DialoGPT-medium
- Parameters: 354,823,168 (354M)
- Architecture: GPT-2 based conversational model

**Knowledge Base:**
- Custom text file containing AI/ML domain information
- Topics: Machine Learning, Deep Learning, NLP, Transformers, BERT, GPT, RAG, Vector Databases, Fine-tuning, LoRA
- Text splitting: RecursiveCharacterTextSplitter (chunk_size=200, overlap=50)
- 9 document chunks created

**Vector Store:**
- FAISS (Facebook AI Similarity Search)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Retrieval: Top-3 most relevant documents per query

**RAG Pipeline:**
1. Query received
2. Retrieve top-k relevant documents from vector store
3. Construct prompt with retrieved context
4. Generate response using language model
5. Return answer with source documents

**Example Query:** "What is Retrieval Augmented Generation?"
- Returns contextual answer based on retrieved knowledge base information

**Part B: Fine-tuning with LoRA**

**Model:** gpt2
- Base parameters: 124,439,808 (124M)

**LoRA Configuration:**
- Task type: Causal Language Modeling
- Rank (r): 8
- Alpha: 32
- Dropout: 0.1
- Target modules: c_attn, c_proj
- Trainable parameters: 811,008 (0.65% of total)

**Training Dataset:**
- 10 question-answer pairs on AI/ML topics
- Format: "Question: ... Answer: ..."
- Maximum sequence length: 128 tokens

**Training Configuration:**
- Epochs: 3
- Batch size: 2
- Gradient accumulation steps: 4
- Warmup steps: 10
- Optimizer: AdamW (default)

**Results:**
- Model successfully fine-tuned on domain-specific dataset
- Test generation on "What is machine learning?" demonstrates learned behavior

**Part C: Model Context Protocol (MCP) Server**

**Purpose:** Bridge AI clients to external data sources through standardized tool interface

**Server Implementation:**
- Protocol: HTTP POST requests
- Port: Auto-detection (8000-8009)
- Threading: Daemon thread for non-blocking operation

**Available Tools:**

**1. get_capital**
- Description: Get the capital city of a country
- Input: country (string)
- Output: capital city name
- Database: 20 countries

**2. get_population**
- Description: Get the population of a country
- Input: country (string)
- Output: population (numeric and formatted)
- Database: 20 countries

**Countries Supported:**
United States, Ukraine, France, Germany, United Kingdom, Japan, China, India, Canada, Australia, Brazil, Russia, Italy, Spain, Mexico, South Korea, Argentina, South Africa

**MCP Server Testing:**
- Direct API calls via requests library
- Tool listing endpoint
- Tool execution endpoint
- JSON request/response format

**Integration with Google Gemini:**
- Model: gemini-2.5-flash
- Query parsing for relevant tool selection
- Tool execution with country extraction
- Natural language response generation
- Example queries:
  - "What's the capital of Ukraine?" → Kyiv
  - "What's the population of the United States?" → 331,900,000

**Device Support:**
- MPS (Metal Performance Shaders) for Apple Silicon
- CUDA for NVIDIA GPUs
- CPU fallback

**Dependencies:**
- transformers
- langchain_community
- sentence-transformers
- datasets
- peft
- faiss-cpu
- accelerate>=0.26.0
- google.generativeai

---

### 5. CLIP_Model_Evaluation.ipynb

**Purpose:** Comprehensive evaluation and architecture analysis of OpenAI's CLIP (Contrastive Language-Image Pre-training) model.

**Model:** openai/clip-vit-base-patch32
- Total parameters: 151,277,313 (151M)
- Vision encoder: Vision Transformer (ViT)
- Text encoder: Transformer
- Projection dimension: 512

**Part 1: Model Architecture Analysis**

**Vision Encoder (ViT):**
- Architecture: Vision Transformer
- Image size: 224×224 pixels
- Patch size: 32×32 pixels
- Number of patches: 49 (7×7 grid)
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Intermediate MLP size: 3,072
- Parameters: 85,054,464 (56.2% of total)

**Patch Embedding:**
- Converts 32×32 patches to 768-dimensional embeddings
- Parameters: 2,359,296

**Position Embedding:**
- Learnable position embeddings for 49 patches
- Parameters: 38,400

**Class Token:**
- Learnable token prepended to sequence
- Parameters: 768

**Transformer Layers (×12):**
- Self-attention with Q, K, V projections (768×768 each)
- Multi-head attention (12 heads)
- MLP with expansion factor 4 (768 → 3,072 → 768)
- Layer normalization before attention and MLP
- Parameters per layer: 7,087,872

**Text Encoder:**
- Architecture: Standard Transformer
- Vocabulary size: 49,408 tokens
- Hidden size: 512
- Number of layers: 12
- Number of attention heads: 8
- Intermediate MLP size: 2,048
- Max sequence length: 77 tokens
- Parameters: 37,828,608 (25.0% of total)

**Token Embedding:**
- Maps tokens to 512-dimensional embeddings
- Parameters: 25,296,896 (16.7% of total)

**Position Embedding:**
- Learnable position embeddings
- Parameters: 39,424

**Transformer Layers (×12):**
- Self-attention with Q, K, V projections (512×512 each)
- Multi-head attention (8 heads)
- MLP with expansion factor 4 (512 → 2,048 → 512)
- Layer normalization before attention and MLP
- Parameters per layer: 3,152,384

**Projection Layers:**
- Visual projection: 768 → 512 (393,216 parameters)
- Text projection: 512 → 512 (262,144 parameters)
- Logit scale: Learnable temperature parameter (1 parameter)

**Parameter Breakdown:**
- Vision encoder layers: 85,054,464 (56.2%)
- Text encoder layers: 37,828,608 (25.0%)
- Token embeddings: 25,296,896 (16.7%)
- Patch embedding: 2,359,296 (1.6%)
- Projections: 655,360 (0.4%)
- Bias parameters: 171,008 (0.11%)

**Part 2: Model Evaluation on ModelvsBaby Dataset**

**Dataset:**
- 8 object categories: airplane, car, chair, cup, dog, donkey, duck, hat
- 5 visual conditions: realistic, geons, silhouettes, blurred, features

**Evaluation Methodology:**
- Zero-shot classification using text prompts: "a photo of a {category}"
- Cosine similarity between image and text embeddings
- L2 normalization of embeddings
- Softmax over similarity scores

**Results:**
- Confusion matrices for each condition
- Accuracy metrics per condition
- Comparison across visual conditions

**Part 3: Comparison with Human Infant Performance**

**Baby Performance (2-year-olds from Figure 3A):**
- Realistic: ~85%
- Silhouettes: ~78%
- Geons: ~58%
- Blurred: ~55%
- Features: ~54%

**CLIP vs Baby Performance Analysis:**
- Quantitative comparison across all conditions
- Identification of conditions where CLIP outperforms or underperforms
- Detailed analysis of performance differences
- Insights into model strengths and limitations

**Key Findings:**
- CLIP demonstrates strong performance on realistic and degraded images
- Comparison reveals interesting differences in abstract shape recognition
- Analysis suggests areas where learned representations differ from developmental learning

**Device Support:**
- MPS (Metal Performance Shaders) for Apple Silicon
- CUDA for NVIDIA GPUs
- CPU fallback

**Dependencies:**
- transformers
- torch
- pandas
- matplotlib
- seaborn
- sklearn
- PIL

---

## Dependencies

### Core Libraries
- Python 3.8+
- PyTorch 2.0+
- torchvision
- transformers (Hugging Face)

### Data Processing
- numpy
- pandas
- PIL (Pillow)
- pathlib

### Machine Learning
- scikit-learn
- scipy
- umap-learn
- peft (Parameter-Efficient Fine-Tuning)
- datasets (Hugging Face)

### Visualization
- matplotlib
- seaborn

### Language Models & RAG
- langchain_community
- sentence-transformers
- faiss-cpu
- google.generativeai

### Utilities
- warnings
- collections
- re
- json
- socket
- threading
- requests

## Installation

```bash
pip install torch torchvision transformers numpy pandas pillow scikit-learn scipy umap-learn matplotlib seaborn peft datasets langchain_community sentence-transformers faiss-cpu accelerate google-generativeai
```

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support or Apple Silicon with MPS
- **RAM:** Minimum 8GB, recommended 16GB+
- **Storage:** 2GB+ for models and datasets

## Device Support

All notebooks automatically detect and utilize the best available device:
1. MPS (Metal Performance Shaders) for Apple Silicon
2. CUDA for NVIDIA GPUs
3. CPU as fallback

## Dataset Structure

```
Data/
├── NeuralNetwork/
│   ├── car/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── duck/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── Clustering/
    ├── realistic/
    ├── features/
    ├── blurred/
    ├── geons/
    └── silhouettes/
```

## Usage

Each notebook can be run independently in Jupyter:

```bash
jupyter notebook Car_Duck_Classification_CNN.ipynb
```

Or via Google Colab using the badge links in the notebooks.

## Key Concepts Implemented

### Computer Vision
- Convolutional Neural Networks (CNNs)
- Image classification
- Data augmentation
- Transfer learning with ResNet
- Vision Transformers (ViT)
- Zero-shot classification with CLIP

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Locally Linear Embedding (LLE)
- Uniform Manifold Approximation and Projection (UMAP)

### Clustering
- K-Means clustering
- Expectation-Maximization (EM) / Gaussian Mixture Models (GMM)
- Silhouette score analysis
- BIC and AIC model selection
- Confusion matrix evaluation

### Natural Language Processing
- Retrieval Augmented Generation (RAG)
- Vector embeddings with sentence-transformers
- FAISS vector database
- Fine-tuning with LoRA
- Parameter-efficient training
- Model Context Protocol (MCP) server
- Tool-augmented language models

### Deep Learning Techniques
- Dropout regularization
- Batch normalization
- Layer normalization
- Attention mechanisms
- Multi-head attention
- Transformer architecture
- Residual connections

## Model Architectures

### Custom CNN (Car vs Duck)
- 3 convolutional layers with increasing filters (32 → 64 → 128)
- MaxPooling after each convolution
- 3 fully connected layers with dropout
- 25.8M parameters

### ResNet-18 (Feature Extraction)
- Pre-trained on ImageNet
- 512-dimensional feature vectors
- Used as frozen feature extractor

### DialoGPT-medium (RAG)
- GPT-2 based conversational model
- 354M parameters
- Fine-tuned with LoRA (811K trainable parameters)

### CLIP ViT-Base-Patch32
- Vision Transformer encoder (151M parameters)
- 12-layer vision transformer (768-dimensional)
- 12-layer text transformer (512-dimensional)
- Contrastive learning objective

## Evaluation Metrics

- **Classification Accuracy:** Percentage of correct predictions
- **Confusion Matrix:** Class-wise prediction analysis
- **Loss Curves:** Training and validation loss over epochs
- **Silhouette Score:** Cluster quality measure
- **BIC/AIC:** Model selection criteria for GMM
- **Variance Explained:** PCA component analysis
- **Zero-shot Accuracy:** CLIP performance without task-specific training

## Visualizations

- Training curves (loss and accuracy)
- Confusion matrices with heatmaps
- 2D scatter plots of reduced dimensions
- Elbow curves for optimal k selection
- Generated synthetic images
- PCA reconstruction comparisons
- Condition-specific performance comparisons

## Research Applications

1. **Object Recognition:** Understanding model performance across different visual conditions
2. **Representation Learning:** Comparing hand-crafted features (PCA) vs learned features (ResNet)
3. **Clustering Analysis:** Evaluating unsupervised learning on image data
4. **Model Interpretability:** Parameter breakdown and architecture analysis
5. **Developmental AI:** Comparing model performance with human infant capabilities
6. **Multimodal Learning:** Image-text alignment with CLIP
7. **Conversational AI:** RAG pipeline for knowledge-grounded responses
8. **Tool-augmented AI:** MCP server for external data integration

## Future Enhancements

- Implement additional CNN architectures (ResNet, EfficientNet)
- Explore other dimensionality reduction techniques (Autoencoders, VAE)
- Add hierarchical clustering methods
- Implement DBSCAN for density-based clustering
- Extend to multi-class classification beyond binary
- Fine-tune CLIP on domain-specific data
- Implement attention visualization for transformers
- Add more sophisticated RAG techniques (HyDE, Multi-Query)
- Expand MCP server with additional tools and APIs

## References

1. OpenAI CLIP: https://github.com/openai/CLIP
2. Hugging Face Transformers: https://huggingface.co/docs/transformers/
3. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
4. scikit-learn: https://scikit-learn.org/stable/
5. FAISS: https://github.com/facebookresearch/faiss
6. LangChain: https://python.langchain.com/
7. LoRA Paper: https://arxiv.org/abs/2106.09685
8. Vision Transformer: https://arxiv.org/abs/2010.11929
9. CLIP Paper: https://arxiv.org/abs/2103.00020
10. ModelvsBaby Dataset: https://osf.io/ba4g2

## License

Academic project for CSCI-P 556 Computer Vision course.

## Author

Sunita Kumari

## Acknowledgments

- Indiana University Bloomington
- CSCI-P 556 Computer Vision Course
- Hugging Face for pre-trained models
- OpenAI for CLIP model
- Meta AI for FAISS library
