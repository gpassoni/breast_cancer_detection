# Breast Cancer Detection using Thermal Imaging and Self-Supervised Learning  

## Introduction  
Breast cancer remains one of the leading causes of death among women worldwide. Early detection plays a crucial role in improving treatment success rates and survival outcomes. Traditional screening techniques, while effective, still face limitations in accessibility, invasiveness, and accuracy.  

In this project, we explore **thermal imaging** combined with **artificial intelligence (AI)** as an alternative, non-invasive approach for breast cancer detection. The main objective is to develop an AI-driven pipeline capable of assisting clinicians in identifying potential tumors at early stages, thus increasing the chances of successful treatment.  

## Technical Overview  
One of the major challenges in medical AI is the **lack of large, annotated datasets**. Building robust supervised models is particularly difficult when reliable labels are scarce.  

To overcome this limitation, we adopted a **Self-Supervised Learning (SSL)** strategy. By leveraging SSL, we can train models to learn meaningful representations of thermal breast images **without requiring labels**, making the approach scalable and data-efficient.  

The pipeline includes several key components:  
- **Segmentation**: Breast regions are segmented using **R2AU-Net** (implementation inspired by [LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)) to remove noise and isolate the relevant areas.  
- **Representation Learning**: We used **VICReg** (a self-supervised method developed by [Meta AI](https://github.com/facebookresearch/vicreg)) to learn robust image embeddings from unlabeled thermal data.  
- **Downstream Tasks**: After pretraining, the learned embeddings are fine-tuned for classification tasks using limited labeled data.  

## Results & Comparison  
Despite the scarcity of annotated thermal datasets, our approach shows performance comparable to **state-of-the-art (SOTA)** studies in the domain. By leveraging SSL, we achieve competitive results while relying on **significantly fewer labeled samples**.  

All downstream evaluations are conducted on **publicly available datasets** such as **DMR-IR**, ensuring transparency and reproducibility of the benchmarking process.  

## Usage  
The repository provides ready-to-use scripts for both **segmentation** and **self-supervised learning**:  

- **Segmentation (R2AU-Net)**:  
  Simply use the training script provided in `/scripts` to train the segmentation model. No modifications are required.  
  - We recommend first running the **hyperparameter tuning script** (integrated with **Weights & Biases**) to adapt the model to your specific dataset.  
  - Once tuned, you can train the model directly on your images.  

- **Self-Supervised Learning (VICReg)**:  
  Due to the **high computational requirements**, SSL pretraining was executed on a cluster. The full VICReg source code is not included here.  
  - To reproduce SSL experiments, download the official VICReg implementation from [Meta AI’s GitHub repository](https://github.com/facebookresearch/vicreg).  
  - The `/vicreg` folder contains **checkpoints (.pth)** and configurations with the best parameters from our runs. These can be used to **reconstruct the trained SSL model** or serve as a strong baseline for your own training and evaluation.  

⚠️ **Note on data availability**:  
- The proprietary thermal images used for SSL pretraining and segmentation cannot be shared.  
- However, all evaluation and testing have been carried out using **public datasets (e.g., DMR-IR)**, which are freely accessible for research purposes.  

## Repository Structure  
- **`/notebooks`** → Jupyter notebooks for visualization of results and downstream evaluations.  
- **`/scripts`** → Code for segmentation (R2AU-Net) and training of classification models.  
- **`/vicreg`** → Checkpoints (.pth) and configurations to reproduce SSL models with the best parameters from our training runs.  
- **`/exploration_and_testing`** → Exploratory notebooks showcasing individual pipeline components (e.g., preprocessing, augmentations, ROI detection).  

## Conclusion  
This project demonstrates the potential of **self-supervised learning on thermal breast images** as a viable tool for early cancer detection. While challenges remain in terms of dataset availability and clinical validation, the results highlight how SSL can reduce the dependency on large labeled datasets while achieving SOTA-level performance.  
