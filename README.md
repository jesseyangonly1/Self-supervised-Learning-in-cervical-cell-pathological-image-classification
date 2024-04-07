# Self-supervised-Learning-in-cervical-cell-pathological-image-classification
This repository is for pathological image classification. Supervised Learning has obtained good results in this task however it requires a large number of annotations from professional doctors, which is time-consuming and costly. Self-supervised Learning attracts extensive attention in the industry because of its characteristics of self-mining information without manual annotation and benefiting almost all downstream tasks. In this repository the Contrastive Learning method (CL) (one of the
Self-supervised Learning methods) was used to pre-train ResNet18, and then the trained model was transferred to the downstream cervical negative and positive cells’ classification task for Fine-tuning evaluation and Linear evaluation. You need to prepare your own dataset, before being input to the net all pictures need to be resize to 224*224.
Code is in master branch.
