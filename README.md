# Customer-Churn-Prediction-using-ANN
This project predicts customer churn (whether a customer will leave the bank) using an Artificial Neural Network (ANN). It demonstrates a complete ML pipeline: preprocessing → training → evaluation → deployment with Streamlit.

### This is how it looks like:
<img width="1919" height="969" alt="image" src="https://github.com/user-attachments/assets/5c24fd4d-3047-408d-8469-bc0dfb0060d5" />


## 📌 Table of Contents
* [Overview](#overview)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Neural Network Architecture](#neural-network-architecture)
* [Model Training](#model-training)
* [Results](#results)
* [Deployment with Streamlit](#deployment-with-streamlit)
* [Repository Structure](#repository-structure)
* [How to Run](#how-to-run)
* [Future Improvements](#future-improvements)
---

## 📖 Overview
Customer churn is a **critical metric** for banks — retaining customers is often more cost-effective than acquiring new ones.
In this project:
* We trained an **ANN model** on the *Churn_Modelling.csv* dataset.
* Achieved ~**86% validation accuracy**.
* Built a **Streamlit app** for real-time churn prediction.
* Tracked training with **TensorBoard** for visualization.
---

## 📊 Dataset
* Source: Kaggle’s **Churn Modelling dataset**.
* Features include:

  * **Demographics**: Age, Gender, Geography
  * **Banking details**: CreditScore, Balance, Tenure, Products, Credit Card usage, etc.
* Target:

  * `Exited` (1 → Customer left, 0 → Customer stayed).
---

## 🛠️ Data Preprocessing
1. **Dropped irrelevant columns**: `RowNumber`, `CustomerId`, `Surname`.
2. **Encoding**:

   * `Gender` → Label Encoding (Male=1, Female=0).
   * `Geography` → One-Hot Encoding (France, Spain, Germany).
3. **Feature Scaling**: StandardScaler to normalize input features.
4. **Pickled preprocessing objects** for reuse in deployment (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`).
---

## 🧠 Neural Network Architecture
The ANN was built using **TensorFlow/Keras**:

```
Input Layer → Dense(128, relu) → Dropout(0.3)  
             → Dense(64, relu) → Dropout(0.3)  
             → Dense(32, relu) → Dropout(0.3)  
             → Dense(1, sigmoid)
```
* **Activation Functions**:
  * ReLU → for hidden layers (handles non-linearity).
  * Sigmoid → for output (binary classification).
* **Dropout**: Prevents overfitting by randomly deactivating neurons.
* **Loss Function**: Binary Crossentropy.
* **Optimizer**: Adam (learning rate = 0.01).
---
### 🔎 Detailed Layer-by-Layer Breakdown

**Input Layer**
* Accepts a flattened array of preprocessed customer features (e.g., CreditScore, Age, Tenure, Balance, Geography, Gender, etc.).

**Dense Layer (dense_5)**
* **Neurons**: 128
* **Activation**: ReLU (Rectified Linear Unit)
* **Purpose**: Learns the first level of complex feature interactions and patterns from the input data.

**Dropout Layer (dropout_3)**
* **Rate**: 0.3 (30%)
* **Purpose**: Regularization to prevent overfitting by randomly deactivating 30% of neurons during training.

**Dense Layer (dense_6)**
* **Neurons**: 64
* **Activation**: ReLU
* **Purpose**: Extracts higher-level abstractions from the patterns learned in the previous layer.

**Dropout Layer (dropout_4)**
* **Rate**: 0.3
* **Purpose**: Further regularization to improve generalization.

**Dense Layer (dense_7)**
* **Neurons**: 32
* **Activation**: ReLU
* **Purpose**: Reduces dimensionality, focusing on the most critical signals for churn prediction.

**Dropout Layer (dropout_5)**
* **Rate**: 0.3
* **Purpose**: Final regularization step before the prediction stage.

**Output Layer (dense_8)**
* **Neurons**: 1
* **Activation**: Sigmoid
* **Purpose**: Outputs a probability between 0 and 1.

  * Close to **1** → Customer likely to churn.
  * Close to **0** → Customer likely to stay.
---

### 📊 Model Summary

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_5 (Dense)             (None, 128)               1,664     
 dropout_3 (Dropout)         (None, 128)               0         
 dense_6 (Dense)             (None, 64)                8,256     
 dropout_4 (Dropout)         (None, 64)                0         
 dense_7 (Dense)             (None, 32)                2,080     
 dropout_5 (Dropout)         (None, 32)                0         
 dense_8 (Dense)             (None, 1)                 33        
=================================================================
Total params: 12,033  
Trainable params: 12,033  
Non-trainable params: 0  
_________________________________________________________________
```

### 🧮 Parameter Calculation (Manual)
The formula for a Dense layer:
[
{Params} = {Input Units} x {Output Units} + {Output Units (biases)}
]
* **dense_5**: ((13 x 128) + 128 = 1,664)
* **dense_6**: ((128 x 64) + 64 = 8,256)
* **dense_7**: ((64 x 32) + 32 = 2,080)
* **dense_8**: ((32 x 1) + 1 = 33)
✅ Total = **12,033 parameters** (matches Keras summary).


## 🏋️ Model Training
* Training/Validation split: **80/20**.
* **Early Stopping**: Stops training if validation loss doesn’t improve for 10 epochs.
* **TensorBoard**: Used to track loss & accuracy curves.

**Sample Training Logs**:
* Epoch 1: val_accuracy = **84.5%**
* Epoch 10: val_accuracy = **85.5%**
* Epoch 20: val_accuracy = **86.1%**

📈 *Accuracy plateaued around 85–86%, which is strong for this dataset.*
---

## 📈 Results
* **Validation Accuracy**: ~86%
* **Validation Loss**: ~0.34

TensorBoard graphs can be found in the `logs/` folder. They visualize:
* Training vs. Validation Loss
* Training vs. Validation Accuracy

#### Evaluation Accuracy vs Iterations Graph:
<img width="1033" height="583" alt="image" src="https://github.com/user-attachments/assets/77eed68a-e453-484f-9a13-cdf4df5ffb31" />

#### Evaluation Loss vs Iterations Graph:
<img width="1059" height="660" alt="image" src="https://github.com/user-attachments/assets/985e67ad-906f-448b-8b28-327929ed6bd2" />

---

## 🌐 Deployment with Streamlit

We created an interactive **web app** (`my_app.py`) where users can input customer details and get churn predictions.

**Steps**:

1. Load model (`model1.h5`).
2. Load encoders & scaler (`.pkl` files).
3. Process user input: encoding + scaling.
4. Predict churn probability.
5. Display result to the user.

**App Demo Flow**:

* User selects **Geography**, **Gender**, **Age**, etc.
* App outputs **Churn Probability** + **Prediction** (Likely / Not Likely to churn).

---

## 📂 Repository Structure

```
customer-churn-prediction-ann/
│
├── notebooks/
│   └── Learning.ipynb
│
├── app/
│   ├── my_app.py
│   ├── model1.h5
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   ├── scaler.pkl
│
├── data/
│   └── Churn_Modelling.csv
│
├── logs/
│
├── requirements.txt
├── README.md
└── .gitignore
```
