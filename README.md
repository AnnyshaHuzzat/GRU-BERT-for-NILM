# GRU-BERT for NILM
This repository contains the implementation of GRU-BERT models for Non-Intrusive Load Monitoring (NILM), as presented in our published work. The models leverage a combination of  GRU layers and BERT-based embeddings to accurately predict household appliance energy consumption and ON/OFF states from aggregated smart meter readings.

# Datasets
The UK Domestic Appliance-Level ELECTRIcity [(UK-DALE)](https://arxiv.org/abs/1404.0284) dataset provides detailed energy usage data from five UK households, including both aggregate mains and individual appliance readings. We use low-frequency data for five appliances namely, refrigerator, washer-dryer, microwave, dishwasher, and kettle. All signals are resampled to six-second intervals with forward-filling applied for gaps under three minutes, ensuring temporal alignment between mains and appliance-level data.
The Reference Energy Disaggregation Dataset [(REDD)](https://github.com/inesylla/energy-disaggregation-DL) is a benchmark NILM dataset widely used for evaluating appliance-level energy disaggregation. For the REDD dataset, we use the pre-processed data where thresholds and parameters are chosen based on experimental settings.

# Training Process
Run the following command to train the model. Dataset, the name of the appliance and number of epochs should be given as input. The hyperparameters can be tuned my modifying the utils.py file.

`python train.py`

# Model Architecture
In this work, we propose a novel improved hybrid deep learning NILM approach based on GRU and BERT, which performs per-appliance prediction of both the power consumption and ON/OFF status of individual appliances. We develop two models that separately combine BERT with unidirectional and bidirectional variants of GRU. Our proposed model is illustrated as follows: 

![Bi-GRU+BERT Model Architecture](biGRU+BERT.png)

# Performance
Our GRU+BERT and Bi-GRU+BERT models boost NILM performance by combining GRU’s temporal learning with BERT’s contextual attention. Bi-GRU+BERT performs best with fewer training epochs, while GRU+BERT delivers more stable results with longer training. On the fridge, both models reached F1 = 0.871 vs. 0.801 for BERT4NILM (≈8.7% gain). For the microwave, GRU+BERT hit 0.553 and Bi-GRU+BERT 0.515, far above the CNN baseline of 0.341 (up to 62% gain). On the washing machine, GRU+BERT achieved 0.877 vs. 0.835 for ELECTRIcity. Overall, Bi-GRU+BERT excels in shorter training, while GRU+BERT provides stronger long-term stability.

# Citation
If you find this work useful, please consider citing our paper:

(Details will be updated soon)

# Acknowledgement

This implementation builds upon the [BERT4NILM](https://github.com/Yueeeeeeee/BERT4NILM/) framework by Yue et al. We sincerely thank the authors for their valuable contribution, which served as the foundation and inspiration for our work.
