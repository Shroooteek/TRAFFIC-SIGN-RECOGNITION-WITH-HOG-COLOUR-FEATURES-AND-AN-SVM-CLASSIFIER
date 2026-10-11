# Traffic Sign Recognition with HOG–Colour Features and an SVM Classifier

A classical (non-deep-learning) computer-vision pipeline that **detects and recognises 43 types of German road signs** in photos and video. Signs are located with HSV colour segmentation, described using **HOG shape + HSV colour features**, compressed with **PCA**, and classified by an **RBF-SVM** (compared against k-NN). The whole system trains in about two minutes on a CPU and every stage can be inspected, which makes it a fast, explainable baseline for traffic sign recognition (TSR).

## Highlights

- **87.2% test accuracy** (macro F1 0.848) across 43 classes on the 12,630 unseen GTSRB test images, versus 68.7% for k-NN.
- **End-to-end:** goes from a raw road scene to labelled bounding boxes with confidence scores, not just cropped-sign classification.
- **Lightweight and explainable:** a 356-value hand-crafted feature vector, no GPU, no large labelled corpus.
- **Works on images and video:** includes a demo that annotates a drive video frame by frame.
- A real *Speed limit 20* photograph is detected and classified at **99% confidence**.

## How it works

```
GTSRB crops + CIFAR-10 backgrounds
        │
        ▼
Training-set building  → ≤200 images/class, 1 augmented copy each, 1,800 background patches (class 43)
        │
Pre-processing         → resize 32×32 → 3×3 Gaussian blur → CLAHE on LAB lightness channel
        │
Feature extraction     → HOG on greyscale (324 values) + HSV histograms 16+8+8 bins (32 values) = 356
        │
Scaling + PCA          → StandardScaler → PCA (95% variance)
        │
Classifier             → k-NN  vs  RBF-SVM (3-fold grid search); best = SVM, C = 5
        │
Full-scene detection   → HSV colour mask (red/blue/yellow) → contour filters → SVM scores each box
                         → keep confidence ≥ 0.60, reject background → non-max suppression (IoU 0.3)
```

**Data**
- [GTSRB](https://huggingface.co/datasets/tanganke/gtsrb) (German Traffic Sign Recognition Benchmark): 26,640 train / 12,630 test images, 43 classes.
- [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10): used only to generate "background" (non-sign) examples, so the detector can reject false candidates.
- Augmentation: rotation (±15°), scale (0.85–1.15), shift (up to 8%), brightness/contrast jitter, blur and noise.

## Results

| Classifier | Best setting | Validation acc. | Test acc. |
|---|---|---|---|
| k-NN | k = 5, distance weights | 85.7% | 68.7% |
| **RBF-SVM** | **C = 5, gamma = scale** | **92.3%** | **87.2%** |

- **Strongest classes (F1):** Yield 0.997, Go straight or right 0.996, Vehicles >3.5t prohibited 0.980, No entry 0.979, Priority road 0.978. Distinct shapes work best.
- **Weakest classes (F1):** Pedestrians 0.342, Beware ice/snow 0.510, Speed limit 20 0.583, Road narrows right 0.623, Traffic signals 0.642. Small digits and fine pictograms are lost at 32×32.
- **Detection:** on a synthetic 4-sign scene, 2 signs were found (Speed limit 100 at 99%, Speed limit 30 at 92%) with no false alarms. Both misses came from the colour-mask/shape-filter stage, not the classifier.

## Getting started

The project is a single Jupyter notebook, developed in **Google Colab**.

1. Open `PRA_mini_proj.ipynb` in Colab (or Jupyter).
2. Install dependencies:
   ```bash
   pip install "opencv-python-headless<5" scikit-learn numpy matplotlib seaborn joblib datasets
   ```
3. Run the cells in order. Datasets download automatically from Hugging Face.
4. The trained pipeline is saved as `traffic_sign_svm.pkl` (loadable with `joblib.load`).

Notebook sections: configuration → data loading → sampling → augmentation → background class → feature extraction → grid search (k-NN & SVM) → final SVM training → test evaluation → colour-based detector → synthetic-scene demo → real-photo demo → video demo.

> The photo and video demos use `google.colab.files.upload()`. Outside Colab, replace it with `cv2.imread(path)` / a local video path.

**Key parameters** (top of the notebook): `IMG_SIZE=32`, `MAX_PER_CLASS=200`, `AUG_COPIES=1`, `N_BACKGROUND=1500`, `CONF_THRESH=0.60`, `RANDOM_STATE=42`.

## Tech stack

Python · OpenCV · scikit-learn · NumPy · Matplotlib · Seaborn · Hugging Face `datasets` · joblib

## Limitations and future work

- Trained on a subset (8,330 crops); using all 26,640 images should help.
- 32×32 crops and hand-crafted features lose fine detail (digits, pictograms).
- The colour mask misses low-contrast or faded signs.
- Next steps: add a small CNN classifier (ResNet reaches ~99.7% on GTSRB), replace the colour mask with a learned detector such as YOLO, and test on full-scene data and embedded hardware (Raspberry Pi / Jetson Nano).


## References

1. J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition," *Neural Networks*, vol. 32, pp. 323–332, 2012.
2. N. Triki, M. Karray, M. Ksantini, "A comprehensive survey and analysis of traffic sign recognition systems with hardware implementation," *IEEE Access*, vol. 12, 2024.
3. B. Meng, W. Shi, "Small traffic sign recognition method based on improved YOLOv7," *Scientific Reports*, vol. 15, 2025.
4. H. Zhang, M. Liang, Y. Wang, "YOLO-BS: a traffic sign detection algorithm based on YOLOv8," *Scientific Reports*, vol. 15, 2025.
5. C. Wang, B. Zheng, C. Li, "Efficient traffic sign recognition using YOLO for intelligent transport systems," *Scientific Reports*, vol. 15, 2025.
6. L. Wang, "Comparative experiment and performance evaluation of three different models of traffic sign recognition system," *ITM Web of Conferences*, vol. 78, 2025.
