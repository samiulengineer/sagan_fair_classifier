# SAGAN: Maximizing Fairness using Semantic Attention Based Generative Adversarial Network

## Introduction

From credit ratings to housing allocation, machine learning models are increasingly used to automate everyday decision making processes. With the growing impact on society, more and more concerns are being voiced about the loss of transparency, accountability and fairness of the algorithms making the decisions. We as data scientists need to step-up our game and look for ways to mitigate emergent discrimination in our models. We need to make sure that our predictions do not disproportionately hurt people with certain sensitive characteristics (e.g., gender, ethnicity).

Luckily, last year's NIPS conference showed that the field is actively investigating how to bring fairness to predictive models. The number of papers published on the topic is rapidly increasing, a signal that fairness is finally being taken seriously. This point is also nicely made in the cartoon below, which was taken from the excellent CS 294: Fairness in Machine Learning course taught at UC Berkley.

![Alternate text](/readme/fairness_plot.svg)

Some approaches focus on interpretability and transparency by allowing deeper interrogation of complex, black box models. Other approaches, make trained models more robust and fair in their predictions by taking the route of constraining and changing the optimization objective.

Here, we will train a model for making income level predictions, analyse the fairness of its predictions and then show how adversarial training can be used to make it fair. The used approach is based on the 2017 NIPS paper "[Learning to Pivot with Adversarial Networks](https://papers.nips.cc/paper/2017/hash/48ab2f9b45957ab574cf005eb8a76760-Abstract.html)" by Louppe et al.

## Dataset

For our experiment we use [Ault UCI](https://archive.ics.uci.edu/ml/datasets/Adult) dataset which can be download from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/). It is also referred to as "Census Income" dataset. Here, we will predict whether or not a person's income is larger than 50K dollar a year. It is not hard to imagine that financial institutions train models on similar data sets and use them to decide whether or not someone is eligible for a loan, or to set the height of an insurance premium. The dataset contain the following features:

![Alternate text](/readme/datasetDes.jpeg)

### Sensitive Features

In the Adult UCI dataset there are two sensitive featues.

- ### race

Distribution of race in the dataset.

![Alternate text](/readme/race%20distribution.png)

- ### sex

Distribution of sex in the dataset.

![Alternate text](/readme/gender%20distribution.png)

### Mutual Information

Mutual information is a measure of non-linear association between two variables. It indicates how much the uncertainty around one variable is reduced by observing another. In the below, you can see the mutual information values between each of the 6 features and protected features. Notice the high value between marital-status and sex. This suggests a possible relationship between these variables. In other words, marital-status could be a proxy variable for sex. This is done using the `mutual_info_classif` function.

![Alternate text](/readme/MI.png)

## Model

Below figure describe our full model and the workflow.

![Alternate text](/readme/SAGAN.png)

## Setup

Use Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

<!-- conda env create -n mlseed -f ./envs/conda_env.yml

conda activate mlseed

cd project -->

## Experiment

After setup the required folders and package run the following command for the experiment.

```
python project/train.py --batch_size 64 \
--iteration 10 \
--test_size 0.2 \
--gpu 2
```

## Result

<img src="readme/best_metrics_new.jpg" alt="drawing" width="1000"/>   |   <img src="readme/best_metrics_new.jpg" alt="drawing" width="1000"/>
:-------------------------:|:-------------------------:
**Result from our experiment**  |  **Previously implementated result**

>**From our experiment, we got the following result**

![Alternate text](/readme/adv.gif)

![Alternate text](/readme/DI%20vs%20AttenWei.jpg)   |   ![Alternate text](/readme/accuracy%20vs%20AttenW.jpg)
:-------------------------:|:-------------------------:
**DI vs Attention Weight**  |  **Accuracy vs Attention Weight**
