# MechSight AI

Streamlit website for whole-image classification of mechanical surface images as Normal (N) or Defective/Pitting (P).

## Files
- `app.py`: website
- `mechanical_defect_detector.keras`: trained model
- `requirements.txt`: Python dependencies

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push these files to a GitHub repository, then deploy `app.py` from Streamlit Community Cloud:
https://share.streamlit.io/

Important: the app performs whole-image classification. It does not locate defects with bounding boxes.
