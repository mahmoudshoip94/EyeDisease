# Eye Disease Classification — Cellula CV Engineer Internship

## Project Overview

This project is an 8-class retinal disease classification system developed as part of the **Cellula CV Engineer Internship task**.

The main challenge in the project was not only model selection, but the quality and distribution of the available training data. The dataset was highly imbalanced, with some classes containing thousands of images while others contained only a very small number of samples.

Because of this, the project was developed through several controlled experiments, starting from a standard multiclass model and progressively trying different strategies to handle the data imbalance.

---

# 1. Dataset Problem

The original dataset contained **11,839 retinal images** collected from several sources:

| Source | Images |
|---|---:|
| ODIR5K | 7,000 |
| APTOS2019 | 3,484 |
| ACRIMA | 705 |
| ORIGA | 650 |
| **Total** | **11,839** |

The dataset was not balanced across the target classes.

### Original class distribution

| Class | Images | Percentage |
|---|---:|---:|
| Normal | 4,698 | 39.68% |
| Diabetic Retinopathy | 4,113 | 34.74% |
| Others | 1,102 | 9.31% |
| Glaucoma | 930 | 7.86% |
| Cataract | 340 | 2.87% |
| Myopia | 294 | 2.48% |
| AMD | 274 | 2.31% |
| Hypertension | 88 | 0.74% |

The most obvious problem was the large difference between the majority and minority classes.

For example:

```text
Normal            → 4698 images
Diabetic Retinopathy → 4113 images
Hypertension      → 88 images
```

This imbalance made it difficult for a single multiclass model to learn all diseases equally well.

Therefore, the project focused heavily on **data preparation, augmentation, class weighting, and alternative classification strategies**.

---

# 2. Data Preparation and Augmentation

Before experimenting with different model architectures, the dataset was split into:

```text
Train      → 70%
Validation → 15%
Test       → 15%
```

The resulting working dataset was organized as:

```text
Data/
├── images/
├── metadata.csv
├── images_augmented_v2/
└── dataset_splits_v2/
    ├── train/
    ├── val/
    └── test/
```

The test and validation sets were kept separate from training augmentation.

### Training augmentation

The training data used augmentation to increase the representation of the minority classes.

The augmentation pipeline included transformations such as:

- Horizontal Flip
- Shift / Scale / Rotation
- Brightness / Contrast
- Blur
- Gaussian Noise

The purpose was to make the model less dependent on the original limited examples of the minority classes.

After fixing the augmentation strategy, it was kept consistent while testing the following modeling approaches.

---

# 3. Experiment 1 — Standard Multiclass Model

## Notebook

```text
Modus
```

The first experiment was the standard multiclass approach.

The model received a retinal image and directly predicted one of the eight target classes:

```text
Input Image
     │
     ▼
Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
CNN / ResNet Model
     │
     ▼
8-Class Softmax
     │
     ├── Normal
     ├── Diabetic Retinopathy
     ├── Glaucoma
     ├── Cataract
     ├── AMD
     ├── Hypertension
     ├── Myopia
     └── Others
```

The first objective was to establish a baseline and understand how the model behaved with the imbalanced dataset.

The experiment showed that the model could learn the majority classes relatively well, but performance on several minority classes was much weaker.

### Confusion Matrix

Replace the placeholder below with the confusion matrix generated from the `Modus` notebook:

![Modus — Confusion Matrix](assets/confusion_matrix_modus.png)

> **Image placeholder:** `assets/confusion_matrix_modus.png`

### Experiment 1 observations

The confusion matrix showed that the model was not treating all classes equally.

The majority classes had substantially more training examples, while minority diseases were more frequently confused with other classes.

This motivated the next experiment: **class weighting**.

---

# 4. Experiment 2 — Standard Model + Class Weights

## Notebook

```text
Modus class weight
```

After establishing the augmented-data baseline, the next experiment kept the augmentation strategy fixed and introduced **class weights**.

The idea was to give more importance to samples belonging to underrepresented classes during training.

Instead of treating every training error equally, the loss function was adjusted so that mistakes on minority classes contributed more to the optimization process.

Conceptually:

```text
Majority class
      │
      ▼
Lower training weight

Minority class
      │
      ▼
Higher training weight
```

The pipeline therefore became:

```text
Input Image
     │
     ▼
Preprocessing
     │
     ▼
Fixed Training Augmentation
     │
     ▼
Class-Weighted Loss
     │
     ▼
CNN / ResNet
     │
     ▼
8-Class Prediction
```

The purpose of this experiment was to determine whether changing the loss contribution of each class could improve minority-class recognition without changing the dataset again.

### Confusion Matrix

Replace the placeholder below with the confusion matrix generated from the `Modus class weight` notebook:

![Modus + Class Weight — Confusion Matrix](assets/confusion_matrix_modus_class_weight.png)

> **Image placeholder:** `assets/confusion_matrix_modus_class_weight.png`

### Experiment 2 observations

This experiment was useful for understanding the effect of class weighting while keeping the augmentation setup unchanged.

The main question was:

> Does giving minority classes a higher loss weight improve their recall and F1 without causing excessive confusion between the classes?

The answer was not sufficient to solve the overall classification problem, so another strategy was investigated.

---

# 5. Experiment 3 — Two-Model Hierarchical Classification Attempt

## Notebook

```text
Modus 2 Modus
```

The next idea was to divide the classification problem into two models instead of asking one model to distinguish all eight classes directly.

The idea was:

### Model 1

Model 1 would distinguish the major classes while grouping the smaller diseases together.

```text
Input
  │
  ▼
Model 1
  │
  ├── Normal
  ├── Diabetic Retinopathy
  ├── Glaucoma
  ├── Others
  └── Minor Diseases
```

The `Minor Diseases` group contained:

```text
Cataract
AMD
Hypertension
Myopia
```

### Model 2

If Model 1 predicted `Minor Diseases`, the image would be passed to Model 2:

```text
Minor Diseases
       │
       ▼
    Model 2
       │
       ├── Cataract
       ├── AMD
       ├── Hypertension
       └── Myopia
```

### Complete pipeline

```text
                         INPUT IMAGE
                              │
                              ▼
                       ┌─────────────┐
                       │   MODEL 1   │
                       │   5-Class   │
                       └──────┬──────┘
                              │
          ┌───────────┬───────┼────────┬──────────────┐
          │           │       │        │              │
        Normal       DR   Glaucoma   Others     Minor Diseases
          │           │       │        │              │
          ▼           ▼       ▼        ▼              ▼
        FINAL       FINAL   FINAL    FINAL       ┌───────────┐
                                                  │  MODEL 2  │
                                                  │  4-Class  │
                                                  └─────┬─────┘
                                                        │
                                     ┌──────────────────┼─────────────────┐
                                     │                  │                 │
                                  Cataract             AMD       Hypertension / Myopia
```

The idea was to reduce the difficulty of the original 8-class problem by separating the minority diseases into a second classification stage.

---

## Why This Approach Was Not Successful

The hierarchical experiment did not produce the expected validation behavior.

The first model struggled to reliably route images into the correct high-level groups, and this routing error then propagated to the second model.

The result was a validation-loss behavior that showed the approach was not providing a stable solution.

In other words:

```text
Wrong Model 1 decision
        │
        ▼
Wrong route
        │
        ▼
Model 2 receives an inappropriate input group
        │
        ▼
Final prediction becomes unreliable
```

This was an important experiment because it showed that simply splitting the problem into two models does not automatically solve the original data problem.

### Confusion Matrix / Results

Replace the placeholder below with the relevant confusion matrix or result visualization from `Modus 2 Modus`:

![Modus 2 Modus — Confusion Matrix](assets/confusion_matrix_modus_2_modus.png)

> **Image placeholder:** `assets/confusion_matrix_modus_2_modus.png`

---

# 6. Final Approach — Two-Stage Disease Classification

After the previous experiments, a different two-stage strategy was tested using:

```text
new_models_1
new_models_2
```

This approach changed the first question completely.

Instead of asking the first model to distinguish multiple disease classes, the first model only answers:

> Is the eye normal or does it contain an abnormality/disease?

---

# 7. New Model 1 — Normal vs Anomaly

## Notebook

```text
new_models_1
```

Model 1 performs binary classification:

```text
Normal
   vs
Anomaly
```

where `Anomaly` contains all seven non-normal classes:

```text
Diabetic Retinopathy
Glaucoma
Cataract
AMD
Hypertension
Myopia
Others
```

### Model 1 pipeline

```text
                    INPUT IMAGE
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                Training Augmentation
                         │
                         ▼
                    ResNet50
                         │
                         ▼
                  Binary Classifier
                         │
              ┌──────────┴──────────┐
              │                     │
            Normal               Anomaly
              │                     │
              ▼                     ▼
           FINAL              Send to Model 2
```

The dataset used for this stage was transformed from the original 8 classes into two groups.

### Binary dataset

```text
Train:
Normal   = 3000
Anomaly  = 9300

Validation:
Normal   = 705
Anomaly  = 1071

Test:
Normal   = 705
Anomaly  = 1071
```

Class weighting was also used to compensate for the difference between the Normal and Anomaly groups.

The resulting Model 1 was able to perform the first-level screening much more effectively than trying to solve the entire 8-class problem immediately.

### Model 1 Confusion Matrix

Replace the placeholder below with the confusion matrix from `new_models_1`:

![New Model 1 — Normal vs Anomaly Confusion Matrix](assets/confusion_matrix_new_model_1.png)

> **Image placeholder:** `assets/confusion_matrix_new_model_1.png`

---

# 8. New Model 2 — Disease Classification

## Notebook

```text
new_models_2
```

Only images identified as `Anomaly` are passed to the second stage.

Model 2 then determines which disease the abnormal image belongs to.

Conceptually:

```text
Anomaly
   │
   ▼
Model 2
   │
   ├── Diabetic Retinopathy
   ├── Glaucoma
   ├── Cataract
   ├── AMD
   ├── Hypertension
   ├── Myopia
   └── Others
```

This changes the original problem from:

```text
8 classes at once
```

into:

```text
Stage 1:
Normal vs Anomaly

Stage 2:
Which disease?
```

---

# 9. Final Complete Pipeline

The final system can be represented as:

```text
                         ┌─────────────────────┐
                         │     INPUT IMAGE     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PREPROCESSING    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ TRAIN AUGMENTATION  │
                         │   (training only)   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      MODEL 1        │
                         │      ResNet50       │
                         │  Normal vs Anomaly  │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
                  Normal                        Anomaly
                     │                             │
                     ▼                             ▼
                  NORMAL                    ┌──────────────┐
                                           │   MODEL 2    │
                                           │    ResNet50  │
                                           │ Disease Class │
                                           └───────┬──────┘
                                                   │
                  ┌────────────────────────────────┼─────────────────────────┐
                  │              │                 │             │             │
                  ▼              ▼                 ▼             ▼             ▼
                DR           Glaucoma           Cataract        AMD       Hypertension
                                                   │
                                                   └───────────────┐
                                                                   │
                                                                   ▼
                                                              Myopia / Others
```

More precisely, the final prediction space remains:

```text
Normal
Diabetic Retinopathy
Glaucoma
Cataract
AMD
Hypertension
Myopia
Others
```

The difference is that the decision is made in two stages rather than by one direct 8-class classifier.

---

# 10. Final Two-Stage Evaluation

The final evaluation should be performed in two ways.

## Stage-level evaluation

Evaluate Model 1 independently:

```text
Normal
vs
Anomaly
```

Then evaluate Model 2 independently on the disease subset:

```text
DR
Glaucoma
Cataract
AMD
Hypertension
Myopia
Others
```

## End-to-end evaluation

The complete pipeline should also be evaluated from the original test images:

```text
Original Test Image
        │
        ▼
     Model 1
        │
        ▼
 Normal / Anomaly
        │
        ▼
 If Anomaly → Model 2
        │
        ▼
 Final 8-Class Prediction
```

This is important because a good Model 1 and a good Model 2 individually do not necessarily guarantee the same performance when combined.

---

# 11. Final Confusion Matrix

Replace the placeholder below with the final **8-class end-to-end confusion matrix**:

![Final Two-Stage Model — 8-Class Confusion Matrix](assets/confusion_matrix_final_8class.png)

> **Image placeholder:** `assets/confusion_matrix_final_8class.png`

---

# 12. Final Results

The final comparison should report:

| Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Standard Model + Augmentation | TBD | TBD | TBD | TBD |
| Standard Model + Class Weights | TBD | TBD | TBD | TBD |
| Two-Model Hierarchical Attempt | TBD | TBD | TBD | TBD |
| New Model 1 — Normal vs Anomaly | TBD | TBD | TBD | TBD |
| New Model 2 — Disease Classification | TBD | TBD | TBD | TBD |
| Final End-to-End Two-Stage System | TBD | TBD | TBD | TBD |

For the final 8-class task, the following should also be included:

- Per-class Precision
- Per-class Recall
- Per-class F1
- Support
- Confusion Matrix
- Macro F1
- Weighted F1
- Overall Accuracy

---

# 13. Main Lessons From the Experiments

The development process followed a progression from a direct multiclass solution toward a two-stage classification system.

### Experiment 1

```text
Standard 8-Class Model
        ↓
Problem: strong class imbalance
```

### Experiment 2

```text
Standard Model
+
Fixed Augmentation
+
Class Weights
        ↓
Attempt to improve minority-class learning
```

### Experiment 3

```text
Model 1:
Major classes + Minor Diseases

        ↓

Model 2:
Minor Diseases only

        ↓

Problem:
Routing errors + unstable validation behavior
```

### Final approach

```text
Model 1:
Normal vs Anomaly

        ↓

Model 2:
Disease classification

        ↓

Final 8-class prediction
```

The final approach was motivated by the structure of the data itself: first separate normal eyes from abnormal eyes, then focus the second classifier entirely on identifying the specific disease.

---

# 14. Project Structure

A clean final project can be organized as:

```text
EyeDisease/
│
├── Data/
│   ├── images/
│   ├── metadata.csv
│   ├── images_augmented_v2/
│   └── dataset_splits_v2/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── standard/
│   ├── class_weight/
│   ├── hierarchical/
│   ├── new_model_1/
│   └── new_model_2/
│
├── results/
│   ├── confusion_matrix_modus.png
│   ├── confusion_matrix_modus_class_weight.png
│   ├── confusion_matrix_modus_2_modus.png
│   ├── confusion_matrix_new_model_1.png
│   └── confusion_matrix_final_8class.png
│
├── notebooks/
│   ├── Modus.ipynb
│   ├── Modus_class_weight.ipynb
│   ├── Modus_2_Modus.ipynb
│   ├── new_models_1.ipynb
│   └── new_models_2.ipynb
│
└── README.md
```

---

# 15. Current Status

```text
[✓] Original dataset investigated
[✓] Class imbalance identified
[✓] Dataset split into train / validation / test
[✓] Training augmentation implemented
[✓] Standard multiclass model tested
[✓] Class-weight experiment tested
[✓] First hierarchical two-model idea tested
[✓] Normal-vs-Anomaly model implemented
[✓] Disease classification model implemented
[✓] Final two-stage pipeline implemented

[ ] Add final confusion-matrix images
[ ] Add final numerical results
[ ] Run final end-to-end evaluation
[ ] Compare all experiments using the same test set
[ ] Finalize the report
```

---

# 16. Important Note

The project went through several experiments because the initial dataset distribution made direct 8-class classification difficult.

The experiments were therefore not independent random models. Each stage was intended to address a specific problem discovered in the previous stage:

```text
Data imbalance
      ↓
Augmentation
      ↓
Class weighting
      ↓
Hierarchical classification attempt
      ↓
Normal vs Anomaly
      ↓
Disease classification
      ↓
Final two-stage pipeline
```

The confusion matrices and final metrics should be inserted into this README after the final evaluation so that the document represents the actual measured behavior of each experiment.
