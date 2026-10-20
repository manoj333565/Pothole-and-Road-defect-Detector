# 🛣️ Road DetectAI — AI-Powered Pothole Detection

Road DetectAI is an AI-powered pothole detection system built using Python, YOLOv8, and Streamlit. It detects potholes in road images, displays bounding boxes and confidence scores, estimates damage severity, and allows users to download annotated images and detection reports.

## ✨ Features

- Pothole detection using YOLOv8
- Bounding boxes and confidence scores
- Severity estimation based on image-area coverage
- Interactive Streamlit dashboard
- Detection count and detailed results table
- Downloadable CSV reports
- Downloadable annotated images
- Adjustable detection confidence threshold

## 🧰 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Streamlit
- Pandas
- Pillow
- OpenCV

## 📁 Project Structure

```text
Road-DetectAI/
├── app.py
├── train.py
├── severity.py
├── requirements.txt
├── README.md
├── .gitignore
├── Pothole.v1-raw.yolov8/
└── runs/
    └── detect/
        └── runs/
            └── pothole/
                └── pothole_img416_fixed/
                    └── weights/
                        └── best.pt
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your GitHub repository URL.

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

Ensure your trained model (`best.pt`) exists at the path configured in `app.py`.

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, upload a road image, adjust the confidence threshold if needed, and click **Run Road Damage Detection**.

## 🧠 Model Training

The project uses YOLOv8n, trained on a pothole dataset.

Training configuration:

- Epochs: 30
- Image size: 416
- Batch size: 2
- Device: CPU

Check the dataset paths in `Pothole.v1-raw.yolov8/data.yaml` before training.

Run training with:

```bash
python train.py
```

## 📊 Model Evaluation

Latest reported validation results on 133 images containing 330 pothole instances:

| Metric | Result |
|---|---:|
| Precision | 76.2% |
| Recall | 63.3% |
| mAP@50 | 72.1% |
| mAP@50–95 | 43.7% |

Performance may vary on unseen images.

## 📐 Severity Estimation

The application estimates damage coverage using:

`Coverage (%) = (Sum of bounding-box areas / Image area) × 100`

Severity levels:

| Coverage | Severity |
|---|---|
| No detections | NONE |
| Less than 3% | LOW |
| 3% to less than 10% | MEDIUM |
| 10% to less than 20% | HIGH |
| 20% or more | CRITICAL |

These are visual estimates, not measurements of real-world pothole dimensions. Overlapping boxes may count the same area more than once.

## ⚠️ Limitations

- Small or poorly lit potholes may be missed.
- Road markings, shadows, and patches may cause false detections.
- Detection confidence is not the same as overall accuracy.
- Severity estimates depend on image framing and bounding-box quality.
- The application does not replace professional road inspection.

## 🔮 Future Improvements

- Improve pothole detection and recall.
- Evaluate a confusion matrix and precision–recall curve.
- Test on more road images and different lighting conditions.
- Add video or live-camera detection.
- Deploy the application to Hugging Face Spaces.

## 📄 Dataset and License

Check the dataset license and attribution requirements before redistributing it. The current dataset configuration identifies the source license as ODbL v1.0. Verify its terms before publishing the dataset.

## 👨‍💻 Author

**Manoj Kumar Mandal**  
GitHub: (https://github.com/manoj333565)

---

⭐ If you find this project useful, consider starring the repository on GitHub.
