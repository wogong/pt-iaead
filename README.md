# Improved Autoencoder for Unsupervised Anomaly Detection

## Introduction
This is the official implementation of the IAEAD framework presented by "Improved Autoencoder for Unsupervised Anomaly Detection
". The codes are used to reproduce experimental results of CAE and IAEAD reported in the paper.

## Requirements
- Python 3.6
- PyTorch 1.3.1 (GPU)
- Keras 2.2.0 
- Tensorflow 1.8.0 (GPU)
- sklearn 0.19.1
 
## Usage

To obtain the results of IAEAD on MNIST with default settings, simply run the following command:

```bash
python main.py --run_times=1 --gpu_id=0 --method=RAE --dataset=mnist --ratio=0.1 --para_lambda=5e-5
```

After training, to print UAD results for a specific algorithm in AUROC/AUPR, run:

```bash
# AUROC of IPAE on CIFAR10 with outlier ratio 0.1
python evaluate_roc_auc.py --dataset cifar10 --algo_name iae-0.1

# AUPR of IPAE on MNIST with outlier ratio 0.25 and inliers as the postive class
python evaluate_pr_auc.py --dataset mnist --algo_name iae-0.1 --postive inliers
```

## Credit

- https://github.com/demonzyj56/E3Outlier
- https://github.com/gilshm/anomaly-detection
- https://github.com/izikgo/AnomalyDetectionTransformations
- https://github.com/bearpaw/pytorch-classification

## License

IAEAD is released under the MIT License.
