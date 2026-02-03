# Variational Autoencoder (VAE) for Anomaly Detection in High-Dimensional Time Series Data

## Project Overview
This project implements a **Variational Autoencoder (VAE)**–based anomaly detection system designed for **high-dimensional data**.  
The primary objective is to leverage the probabilistic latent space of VAEs to identify anomalous observations that deviate significantly from learned normal patterns.

The project strictly follows advanced anomaly detection requirements, including:
- Synthetic data generation with embedded anomalies
- VAE implementation with ELBO optimization
- Latent space–aware anomaly scoring
- Quantitative evaluation using AUC-based metrics

---

## Dataset Generation
A synthetic high-dimensional dataset is programmatically generated to simulate complex system behavior.

**Dataset characteristics:**
- Number of samples: 10,000  
- Number of features: 100  
- Anomaly ratio: 5%  
- Normal samples: drawn from a standard normal distribution  
- Anomalies: injected as large-magnitude deviations across multiple features  

Only **normal samples** are used during training to ensure unsupervised anomaly detection.

---

## Data Preprocessing
All features are standardized using **z-score normalization** via `StandardScaler`.  
This ensures:
- Stable VAE training
- Balanced contribution of all features
- Improved convergence behavior

---

## Model Architecture
The Variational Autoencoder consists of three main components:

### Encoder
- Fully connected neural network
- Projects input data into a lower-dimensional latent space
- Outputs:
  - Mean vector (μ)
  - Log-variance vector (log σ²)

### Latent Space
- Stochastic latent sampling via the **reparameterization trick**
- Enables backpropagation through probabilistic nodes
- Enforces a Gaussian prior on latent representations

### Decoder
- Reconstructs the input data from the latent representation
- Symmetric architecture to the encoder

---

## Loss Function (ELBO)
The model is optimized using the **Evidence Lower Bound (ELBO)** objective:

ELBO = Reconstruction Loss + β × KL Divergence

Where:
- Reconstruction Loss: Mean Squared Error (MSE)
- KL Divergence: Regularizes the latent distribution toward N(0, I)
- β > 1 introduces **Beta-VAE regularization** to encourage disentangled representations

This balance controls the trade-off between reconstruction fidelity and latent space regularization.

---

## Training Strategy
- Training is performed **only on normal data**
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 20
- Batch size: 64

This setup ensures the VAE learns a compact representation of normal behavior.

---

## Anomaly Scoring Mechanism
An anomaly score is computed using a **hybrid strategy**:

1. **Reconstruction Error**  
   Measures how poorly the VAE reconstructs an input sample.

2. **Latent Space Distance**  
   Computes the squared distance of the latent mean vector from the origin, acting as a Mahalanobis-style regularity measure.

Final anomaly score:
Score = Reconstruction Error + 0.1 × Latent Distance

This combined metric captures both surface-level reconstruction failures and deeper latent deviations.

---

## Evaluation Metrics
The model is evaluated on a test set containing both normal and anomalous samples.

Reported metrics:
- **ROC-AUC**: Measures ranking quality of anomaly scores
- **PR-AUC**: Evaluates precision–recall trade-offs under class imbalance

These metrics are well-suited for anomaly detection problems with rare anomalies.

---

## Results Summary
The trained VAE achieves strong discrimination between normal and anomalous samples, as evidenced by high ROC-AUC and PR-AUC scores.  
This confirms that the learned latent space effectively captures normal behavior while isolating anomalous patterns.

---

## Key Challenges and Trade-offs
- Increasing β improves latent regularization but can degrade reconstruction quality
- High-dimensional inputs require careful normalization to avoid unstable training
- Combining reconstruction and latent scores provides more robust anomaly detection than either alone

---

## Technologies Used
- Python
- PyTorch
- NumPy
- scikit-learn

---

## Project Status
Completed  
Fully documented  
Evaluation-ready  

---

## License
This project is intended for educational and demonstration purposes only.
# vae-autoencoders_anomaly-detection
