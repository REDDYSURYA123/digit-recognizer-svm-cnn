# Digit Recognizer (SVM vs CNN)            [https://bit.ly/4kRVinV]

A high-performance, client-side web application that recognizes handwritten digits and multi-digit numbers directly in the browser. Powered by two distinct machine learning models trained on the **MNIST dataset** with zero backend requirements.

[Digit Recognizer Banner]([public/pwa-512x512.png](https://github.com/REDDYSURYA123/digit-recognizer-svm-cnn/blob/main/public/pwa-512x512.png))

---

## 🚀 Key Features

* **Dual Model Architecture**:
  * **Model 1 (SVM)**: Plain-JavaScript One-vs-Rest Support Vector Machine (using Platt's Simplified SMO algorithm) running inside a non-blocking Web Worker with optional 784→50 PCA dimensionality reduction.
  * **Model 2 (CNN)**: Deep Convolutional Neural Network built with **TensorFlow.js** (`Conv2D → MaxPool → Conv2D → MaxPool → Dropout → Dense`), training live with per-epoch loss and accuracy curves.
* **Multi-Digit Number Recognition (0–100)**: Draw numbers on a responsive 560×280 canvas. The app segments connected components, orders digits left-to-right, applies LeCun center-of-mass preprocessing, and classifies them.
* **Ensemble Mode**: Combine SVM decision scores and CNN probabilities for enhanced overall accuracy.
* **Accuracy-Guarded Feedback Loop**: Correct misclassifications on the fly. The app securely retrains models locally (capped at 10% augmentation) with a strict $\ge 90\%$ accuracy safeguard and automatic rollback.
* **Progressive Web App (PWA)**: Fully responsive layout for mobile and desktop, working entirely offline after the first load.

---

## 🛠️ Tech Stack

* **Frontend**: React, Tailwind CSS, TypeScript, Vite, Lucide Icons
* **Machine Learning**: Vanilla JavaScript SMO SVM (Custom Web Worker), TensorFlow.js 4.22.0
* **Storage**: IndexedDB (for CNN weights and model snapshots), localStorage (for hyperparameters and settings)

---

## 📦 Project Setup & Local Development

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/REDDYSURYA123/digit-recognizer-svm-cnn.git](https://github.com/REDDYSURYA123/digit-recognizer-svm-cnn.git)
   cd digit-recognizer-svm-cnn
