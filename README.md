# Parkinson's Disease Finger-Tapping Severity Classifier
### COS5031-E | Group 25 — Team Yr2DSP | University of Bradford

> An AI and signal processing system that quantifies Parkinson's Disease motor symptom severity from finger-tapping data, using Random Forest and LSTM classification against MDS-UPDRS Item 3.4 scoring criteria.

---

## The Problem

The standard clinical assessment for Parkinson's Disease motor symptoms — the finger-tapping test — relies on subjective visual observation by a neurologist using the MDS-UPDRS scale. This introduces inter-rater variability, is time-intensive, and is difficult to replicate consistently across NHS settings (Goetz et al., 2008).

This project addresses that gap with an objective, data-driven classification system.

---

## What This System Does

1. Takes finger-tapping signal data as input (amplitude, velocity, rhythm time-series)
2. Cleans and normalises signals using IQR outlier removal and Savitzky-Golay smoothing
3. Extracts 17 quantitative features per recording
4. Classifies motor severity as **Normal / Mild / Moderate / Severe**
5. Outputs a confidence score and feature importance explanation

**Target performance:** F1-score ≥ 0.75 (PID Objective 2)  
**Achieved:** Random Forest F1 = 0.8988 | LSTM F1 = 0.7574 ✅

---

## Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Abdul Moeed Alam | Project Lead / Scrum Master | LSTM architecture, model training, GitHub, sprint management |
| Toufiq Rifat | Technical Lead / Data Scientist | Signal processing pipeline, feature extraction, data preprocessing |
| Pierre Andoulo | Product & Ethics Lead | PID, ethics framework, Streamlit app, wireframe design, GDPR compliance |

**Supervisor:** Dr. Ramzi — University of Bradford  
**Advisor:** Dr. Kulvinder Panesar  
**Client:** Kieran Townsend — Future AI for ALL (FALL)

---

## Repository Structure

```
parkinsons-detection-project/
│
├── data/
│   └── raw/
│       └── finger_tapping_features.csv   # 590 recordings, 17 features
│
├── src/
│   └── models/
│       ├── train_lstm_skeleton.ipynb     # LSTM model (Moeed)
│       └── PD_FingertapDemo_v2.ipynb    # Full demo notebook
│
├── app.py                                # Streamlit web application
├── test_plan.py                          # 31-test automated test suite
├── requirements.txt                      # Python dependencies
├── PID_ParkinsonAI_Group25.docx         # Project Initiation Document
├── Ethics_Legal_Framework_Group25.docx  # Ethics & Legal Framework
└── README.md
```

---

## Quick Start

### Run the demo notebook
Open `src/models/PD_FingertapDemo_v2.ipynb` in Kaggle or Jupyter and run all cells.  
Upload `data/raw/finger_tapping_features.csv` when prompted.

### Run the Streamlit app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then upload `finger_tapping_features.csv` in the browser interface.

### Run the test suite
```bash
python test_plan.py
```
Expected output: 31 tests passed, 0 failures.

---

## Technical Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core language |
| TensorFlow / Keras | 2.13 | LSTM model |
| scikit-learn | 1.3.0 | Random Forest, SVM, metrics |
| SciPy | 1.11.2 | Savitzky-Golay signal smoothing |
| Pandas / NumPy | latest | Data handling |
| Streamlit | latest | Web application |
| Matplotlib / Seaborn | latest | Visualisation |

---

## Models

| Model | F1 (weighted) | Notes |
|-------|--------------|-------|
| Random Forest | 0.8988 ± 0.0121 | Primary model — 5-fold CV |
| SVM (RBF) | 0.5558 ± 0.0541 | Baseline comparison |
| LSTM | 0.7574 | Deep learning extension |

All models use stratified k-fold cross-validation following Lones (2024) best practice.  
Training parameters are calculated on training folds only to prevent data leakage.

---

## Ethics & Compliance

- **GDPR:** UK GDPR Articles 6(1)(e) and 9(2)(j) — pre-approved anonymised dataset
- **Anonymisation:** 5-step pipeline (facial suppression, ID removal, metadata stripping, age banding, dataset separation)
- **Safe AI:** Decision-support tool only — not a diagnostic device. All outputs require neurologist review
- **Bias mitigation:** Stratified cross-validation, balanced class weights, subgroup analysis
- **Ethical toolkit:** REFORMS checklist (Kapoor et al., 2024) — 18 items, fully documented
- **Regulatory:** Aligned with MHRA SaMD guidance and NHS AI Lab ATRS

> ⚕️ **Disclaimer:** This system is a clinical decision-support aid only. All AI-generated outputs must be reviewed by a qualified neurologist before any clinical action is taken. This prototype is not MHRA-registered and cannot be used for actual patient diagnosis.

---

## Reproducing Results

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `finger_tapping_features.csv` in `data/raw/`
4. Run `python test_plan.py` to verify the environment
5. Open `src/models/PD_FingertapDemo_v2.ipynb` and run all cells

Full reproduction instructions are documented in the FAIR report (Appendix of Ethics document).

## PowerPoint:
https://1drv.ms/p/c/e88e57c68415e225/IQDu_BvpqqQVRIBNNSeHIULhAVreOOz0kjpJJvrrMUBcLA8

## Demo Video
[Watch the demo]
https://drive.google.com/file/d/1UuupjVsxX4tc0PH6zrKGbuYfHlCucMEq/view?usp=sharing
---

## References

- Goetz et al. (2008) MDS-UPDRS. *Movement Disorders*, 23(15), 2129–2170.
- Kapoor et al. (2024) REFORMS checklist. *Science Advances*, 10, eadk3452.
- Lones (2024) Avoiding common ML pitfalls. *Patterns*, 5(10), 101046.
- Parkinson's UK (2022) The Parkinson's Prevalence Project.
- ICO (2023) Guide to the UK GDPR.

---

*University of Bradford — COS5031-E AI Project Design and Development — April 2026*
