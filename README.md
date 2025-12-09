# Face Identification Demo (Streamlit)

Interactive viewer for precomputed four-choice identification trials using sketches, descriptions, and pose-augmented galleries.

## Run

1. (Optional) Create/activate an environment.
2. Install deps:

```pwsh
pip install -r requirements.txt
```

3. Launch the app:

```pwsh
streamlit run app/streamlit_app.py
```

Then open the URL shown in the terminal.

## What it shows
- Select model (FaceLLM-8B/38B, InternVL2), pose, crop variant, and query modality.
- Displays accuracy and lets you browse individual trials.
- Visualizes the query sketch and the four candidate images with correctness.

## Notes
- The result JSONs reference remote filepaths; the app maps basenames to local folders: `final_dataset_sketches/`, `final_dataset_images/`, `final_dataset_images_croped/`, `poses_all/`, `poses_all_croped/`.
- If an image isn’t found locally, a warning is shown next to that candidate.
