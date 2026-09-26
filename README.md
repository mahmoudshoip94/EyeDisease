# Eye Disease Classification

## Overview

This project focuses on multi-class retinal disease classification using deep learning.

The main challenge encountered during the project was the **highly imbalanced and difficult nature of the dataset**. Some classes contained thousands of images, while other diseases had very few samples. This made direct 8-class classification difficult, especially for minority diseases.

The project therefore went through several experiments to investigate different strategies for handling the data imbalance and improving disease classification.

The final approach evolved from direct 8-class classification into a **two-stage pipeline**:

```text
Input Image
     │
     ▼
 Model 1
Normal vs Anomaly
     │
     └── Anomaly
            │
            ▼
         Model 2
     Disease Classification
Dataset

The dataset contains 11,839 retinal images collected from multiple sources:

Source	Images
ODIR5K	7,000
APTOS2019	3,484
ACRIMA	705
ORIGA	650
Total	11,839

The target classes are:

Normal
Diabetic Retinopathy
Glaucoma
Cataract
AMD
Hypertension
Myopia
Others
Original Class Distribution
Class	Images	Percentage
Normal	4,698	39.68%
Diabetic Retinopathy	4,113	34.74%
Others	1,102	9.31%
Glaucoma	930	7.86%
Cataract	340	2.87%
Myopia	294	2.48%
AMD	274	2.31%
Hypertension	88	0.74%

The large difference between the majority and minority classes was the main motivation for experimenting with augmentation, class weighting, and alternative classification strategies.

1. Only Augmentation
Notebook
models.ipynb

The first experiment used the standard 8-class classification setup.

The model was trained to directly predict one of the eight target classes:

Input Image
     │
     ▼
Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
Classification Model
     │
     ▼
8-Class Prediction

The main strategy used in this experiment was data augmentation, especially to increase the representation of minority classes.

The augmentation pipeline included:

Horizontal Flip
Shift / Scale / Rotation
Brightness / Contrast
Blur
Gaussian Noise

The dataset was split into:

Train      → 70%
Validation → 15%
Test       → 15%

The augmentation was applied only to the training data.

The objective of this experiment was to establish a baseline and investigate how well a direct 8-class classifier could perform after balancing the training data through augmentation.

Results

The model achieved:

Accuracy       : 0.6284
Macro Precision: 0.4756
Macro Recall   : 0.4870
Macro F1       : 0.4567
Weighted F1    : 0.6006
Classification Report
Class	Precision	Recall	F1-Score	Support
AMD	0.2500	0.1220	0.1639	41
Cataract	0.4194	0.7647	0.5417	51
Diabetic Retinopathy	0.6547	0.7407	0.6951	617
Glaucoma	0.5902	0.5143	0.5496	140
Hypertension	0.4000	0.1538	0.2222	13
Myopia	0.6552	0.8636	0.7451	44
Normal	0.6631	0.7064	0.6841	705
Others	0.1724	0.0303	0.0515	165
Confusion Matrices
CNN

ResNet

The confusion matrices demonstrate the difficulty of directly separating all eight classes, particularly the minority classes and the Others class.

2. Class Weights
Notebook
models_ClassWeights.ipynb

After the augmentation-based experiment, the next strategy was to keep the augmentation setup and introduce class weights.

Instead of changing the dataset again, class weights were used to give minority classes a higher contribution to the training loss.

The idea was:

Majority Classes
       ↓
Lower Weight

Minority Classes
       ↓
Higher Weight

The training pipeline therefore became:

Input Image
     │
     ▼
Data Augmentation
     │
     ▼
Class-Weighted Loss
     │
     ▼
Classification Model
     │
     ▼
8-Class Prediction

The goal was to determine whether class weighting could improve minority-class recognition.

Results

The class-weight experiment produced:

Test Loss       : 4.3096
Test Accuracy   : 0.3970
Macro Precision : 0.0496
Macro Recall    : 0.1250
Macro F1        : 0.0710
Weighted F1     : 0.2256
Classification Report
Class	Precision	Recall	F1-Score	Support
AMD	0.0000	0.0000	0.0000	41
Cataract	0.0000	0.0000	0.0000	51
Diabetic Retinopathy	0.0000	0.0000	0.0000	617
Glaucoma	0.0000	0.0000	0.0000	140
Hypertension	0.0000	0.0000	0.0000	13
Myopia	0.0000	0.0000	0.0000	44
Normal	0.3970	1.0000	0.5683	705
Others	0.0000	0.0000	0.0000	165

The model effectively collapsed its predictions toward the Normal class, resulting in very poor performance across the remaining classes.

Confusion Matrix

This experiment showed that class weighting alone did not provide a reliable solution for the imbalance and classification difficulty in this dataset.

3. Hierarchical Model
Notebook
models_2Models.ipynb

The next idea was to divide the original 8-class problem into two classification stages.

Instead of asking one model to distinguish all eight diseases directly, the first model grouped the smaller disease classes together.

Model 1 — 5 Classes

The first model classified:

Normal
Diabetic Retinopathy
Glaucoma
Others
Minor_Diseases

where:

Minor_Diseases =
    Cataract
    AMD
    Hypertension
    Myopia

The pipeline was:

                     Input Image
                         │
                         ▼
                     Model 1
                     5 Classes
                         │
         ┌───────────────┼───────────────┬───────────┐
         │               │               │           │
       Normal            DR           Glaucoma      Others
                                                         
                         Minor_Diseases
                              │
                              ▼
                           Model 2
                           4 Classes
                              │
                    ┌─────────┼─────────┬──────────┐
                    │         │         │          │
                 Cataract     AMD   Hypertension  Myopia
Model 1 Results

The first model achieved:

Accuracy        : 0.5450
Macro Precision : 0.6321
Macro Recall    : 0.6245
Macro F1        : 0.5606
Classification Report
Class	Precision	Recall	F1-Score	Support
Normal	0.8920	0.4922	0.6344	705
Diabetic Retinopathy	0.9040	0.4733	0.6213	617
Glaucoma	0.6078	0.6643	0.6348	140
Others	0.1887	0.8485	0.3087	165
Minor_Diseases	0.5680	0.6443	0.6038	149
Confusion Matrix

Why the Hierarchical Approach Was Not Completed

The second stage of the hierarchical approach, which was supposed to classify the Minor_Diseases, did not train successfully.

The validation loss became:

val_loss = NaN

This made the second-stage model unreliable and prevented the complete hierarchical pipeline from being used as the final solution.

Therefore, this approach was treated as an unsuccessful experiment rather than the final model.

4. Binary + Multi-Class Classification

After the previous experiments, the problem was reformulated into a two-stage classification pipeline.

Instead of trying to classify all eight classes directly, the system first determines whether the eye is normal or abnormal.

Then, if the eye is abnormal, a second model determines the specific disease.

                     Input Image
                         │
                         ▼
                 ┌─────────────────┐
                 │     Model 1     │
                 │ Normal/Anomaly  │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │                 │
               Normal           Anomaly
                 │                 │
                 ▼                 ▼
              NORMAL        ┌───────────────┐
                            │    Model 2    │
                            │ Disease Class │
                            └───────┬───────┘
                                    │
                                    ▼
                         Disease Classification
Model 1 — Normal vs Anomaly
Notebook
new_models_1.ipynb

The first model performs binary classification:

Normal
   vs
Anomaly

where Anomaly represents all non-normal cases.

The anomaly group contains:

Diabetic Retinopathy
Glaucoma
Cataract
AMD
Hypertension
Myopia
Others
Results
Accuracy          : 0.7241

Anomaly Precision : 0.6964
Anomaly Recall    : 0.9617
Anomaly F1        : 0.8078

Macro Precision   : 0.7792
Macro Recall      : 0.6624
Macro F1          : 0.6594

Weighted Precision: 0.7621
Weighted Recall   : 0.7241
Weighted F1       : 0.6900
Classification Report
Class	Precision	Recall	F1-Score	Support
Normal	0.8620	0.3631	0.5110	705
Anomaly	0.6964	0.9617	0.8078	1071
Accuracy

Confusion Matrix

This model was used as the first screening stage of the final pipeline.

Model 2 — Disease Classification
Notebook
new_models_2.ipynb

The second model receives the abnormal/diseased cases and performs multi-class disease classification.

For this stage, the Normal and Others classes were removed.

The model therefore focuses on distinguishing between the actual named diseases:

Diabetic Retinopathy
Glaucoma
Cataract
AMD
Hypertension
Myopia

The resulting dataset distribution is shown below.

This visualization shows the distribution of the remaining disease classes across the training, validation, and test sets after removing Normal and Others.

Results

The second model achieved:

Accuracy        : 0.75
Macro Precision : 0.56
Macro Recall    : 0.70
Macro F1        : 0.60
Weighted F1     : 0.78
Classification Report
Class	Precision	Recall	F1-Score	Support
Diabetic Retinopathy	0.94	0.74	0.83	617
Glaucoma	0.71	0.81	0.76	140
Cataract	0.62	0.84	0.72	51
AMD	0.24	0.56	0.34	41
Hypertension	0.10	0.31	0.15	13
Myopia	0.73	0.91	0.81	44

The model performed particularly well on:

Diabetic Retinopathy
Glaucoma
Cataract
Myopia

while the very small classes, especially Hypertension and AMD, remained more difficult.

Accuracy

Confusion Matrix

Final Pipeline

The final system combines the two models into a single inference pipeline.

                         ┌─────────────────┐
                         │   Input Image   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Model 1     │
                         │ Normal/Anomaly  │
                         └────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
                  Normal                    Anomaly
                     │                         │
                     ▼                         ▼
                  NORMAL                  ┌─────────┐
                                          │ Model 2 │
                                          └────┬────┘
                                               │
                                               ▼
                                      Disease Classification
                                               │
                          ┌────────────┬────────┼────────┬───────────┐
                          │            │        │        │           │
                          ▼            ▼        ▼        ▼           ▼
                         DR        Glaucoma  Cataract   AMD    Hypertension
                                                                          │
                                                                          ▼
                                                                       Myopia

The final decision process is therefore:

Step 1:
Is the eye Normal or Anomaly?

        ↓

If Normal:
    → Normal

If Anomaly:
    → Send image to Model 2

        ↓

Step 2:
Which disease?

    → Diabetic Retinopathy
    → Glaucoma
    → Cataract
    → AMD
    → Hypertension
    → Myopia

Note: Model 1 and Model 2 were evaluated as separate stages. The reported 72.41% and 75.00% accuracies are stage-level metrics, not an end-to-end 8-class accuracy.

This approach reduces the initial classification problem from an 8-class decision into two more focused decisions:

Stage 1:
Normal vs Anomaly

Stage 2:
Disease vs Disease
Project Structure
EyeDisease/
│
├── Data/
│
├── docs/
│   ├── class_distribution.png
│   ├── class_w_cm.png
│   ├── cnn_cm.jpeg
│   ├── model1_acc.jpeg
│   ├── model1_cm.jpeg
│   ├── model1_hir_cm.png
│   ├── model2_cm.jpeg
│   ├── models2_acc.jpeg
│   └── resnet_cm.jpeg
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── models.ipynb
│   ├── models_ClassWeights.ipynb
│   ├── models_2Models.ipynb
│   ├── new_models_1.ipynb
│   ├── new_models_2.ipynb
│   └── splits.ipynb
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
Experiment Summary
Experiment	Strategy	Main Result
Only Augmentation	Direct 8-class classification	Accuracy: 62.84%, Macro F1: 45.67%
Class Weights	Augmentation + class-weighted loss	Accuracy: 39.70%, Macro F1: 7.10%
Hierarchical Model	5-class Model 1 + minority Model 2	Model 1 Accuracy: 54.50%, second stage failed with NaN validation loss
Binary + Multi-Class	Normal/Anomaly → Disease classification	Model 1 Accuracy: 72.41%, Model 2 Accuracy: 75.00%
Conclusion

The experiments showed that the main difficulty was strongly related to the structure and imbalance of the dataset.

The direct 8-class approach using augmentation provided a useful baseline, but minority classes remained difficult to classify.

Adding class weights alone did not solve the problem and resulted in a significant drop in performance.

A hierarchical approach was then investigated by grouping minority diseases together, but the second-stage model failed to train properly because its validation loss became NaN.

The final strategy reformulated the task into two stages:

Normal vs Anomaly
        ↓
Disease Classification

This produced a more focused classification pipeline, with the first model handling the normal/abnormal decision and the second model specializing in distinguishing between diseases.

The final architecture therefore reflects an iterative approach where each experiment was motivated by a limitation observed in the previous one.