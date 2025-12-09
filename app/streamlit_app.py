import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import pandas as pd


# ---- Config ----
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIRS = {
    "sketches": WORKSPACE / "final_dataset_sketches",
    "images_uncrop": WORKSPACE / "final_dataset_images",
    "images_crop": WORKSPACE / "final_dataset_images_croped",
    "poses_uncrop": WORKSPACE / "poses_all",
    "poses_crop": WORKSPACE / "poses_all_croped",
}

RESULTS_DIR = WORKSPACE / "results"


# ---- Helpers ----
@st.cache_data(show_spinner=False)
def build_file_index() -> Dict[str, List[Path]]:
    """Map basename -> list[Path] across crop/uncrop & pose folders.

    Ensures we can choose crop-specific variant when requested.
    """
    index: Dict[str, List[Path]] = {}
    for key in ["images_uncrop", "images_crop", "poses_uncrop", "poses_crop"]:
        root = DATA_DIRS.get(key)
        if not root or not root.exists():
            continue
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            for f in root.rglob(pattern):
                index.setdefault(f.name, []).append(f)
    return index

FILE_INDEX = build_file_index()

def _prefer_path(paths: List[Path], crop: bool) -> Path:
    if crop:
        for p in paths:
            sp = str(p).lower()
            if any(token in sp for token in ["images_crop", "images_croped", "poses_all_crop", "poses_all_croped", "poses_crop"]):
                return p
    else:
        for p in paths:
            if "final_dataset_images" in str(p) and "croped" not in str(p):
                return p
        for p in paths:
            if "poses_all" in str(p) and "croped" not in str(p):
                return p
    return paths[0]

def find_local_image(basename: str, crop: bool) -> Optional[Path]:
    paths = FILE_INDEX.get(basename)
    if not paths:
        return None
    return _prefer_path(paths, crop=crop)

def fallback_search_similar(basename: str, crop: bool) -> Optional[Path]:
    """Search for visually similar file when exact basename missing.

    Strategy: strip pose segment '_p_' and extension, then look for any file starting with identity prefix and containing '_fa'.
    """
    stem = Path(basename).stem
    identity = stem.split('_')[0]
    candidates = [name for name in FILE_INDEX.keys() if name.startswith(identity + "_") and "_fa" in name]
    if not candidates:
        return None
    grouped: List[Path] = []
    for name in candidates:
        grouped.extend(FILE_INDEX.get(name, []))
    if not grouped:
        return None
    if crop:
        crop_matches = [p for p in grouped if "crop" in str(p)]
        if crop_matches:
            return crop_matches[0]
    else:
        uncrop_matches = [p for p in grouped if "croped" not in str(p)]
        if uncrop_matches:
            return uncrop_matches[0]
    return grouped[0]

def search_pose_dirs(basename: str, crop: bool) -> Optional[Path]:
    """Direct search within pose subdirectories to catch unindexed naming variants."""
    pose_root = DATA_DIRS["poses_crop"] if crop else DATA_DIRS["poses_uncrop"]
    if not pose_root.exists():
        return None
    for sub in pose_root.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / basename
        if candidate.exists():
            return candidate
    return None


def map_remote_to_local(path_str: str, crop: bool, is_sketch: bool) -> Optional[Path]:
    """Map remote absolute dataset paths from result JSONs to local workspace paths.

    - For sketches: map to final_dataset_sketches/<basename>
    - For images: search across images_(crop|uncrop) and poses_(crop|uncrop)
    """
    basename = os.path.basename(path_str)
    if is_sketch:
        p = DATA_DIRS["sketches"] / basename
        return p if p.exists() else None
    return find_local_image(basename, crop=crop)


def load_results_file(model: str, crop: bool, pose: str, modality: str) -> Tuple[Optional[Path], Optional[List[dict]]]:
    """Resolve a result JSON path based on selections and load it.

    File naming patterns observed:
    - facellm_*: dataset_4_poses_{crop|uncrop}_{pose}_output_f_{modality}.json (pose='0' for frontal)
    - internvl2 (frontal): dataset_4_{crop|uncrop}_output_{modality}.json
    - internvl2 (poses):   dataset_4_poses_{crop|uncrop}_{pose}_output_{modality}.json
    modality in {s, d, s_d}; facellm uses prefix 'f_' before modality.
    """
    model_dir = RESULTS_DIR / model
    crop_key = "crop" if crop else "uncrop"

    def try_load(path: Path) -> Tuple[Optional[Path], Optional[List[dict]]]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return path, data
            except Exception as e:
                st.error(f"Failed to read {path.name}: {e}")
                return path, None
        return None, None

    if model.startswith("facellm"):
        pose_token = "0" if pose == "Frontal" else pose
        filename = f"dataset_4_poses_{crop_key}_{pose_token}_output_f_{modality}.json"
        return try_load(model_dir / filename)
    else:  # internvl2
        # Observed patterns:
        # Frontal uncrop: dataset_4_uncrop_output_s.json / _s_d.json (no d file)
        # Frontal crop description-only uses dataset_4_crop_0_output_d.json (0 token) not poses_ pattern
        # Pose crop description-only uses dataset_4_crop_<pose>_output_d.json (without 'poses')
        # Pose s / s_d use dataset_4_poses_<crop>_<pose>_output_<modality>.json
        pose_token = "0" if pose == "Frontal" else pose
        if modality == "d" and crop and pose == "Frontal":
            filename = f"dataset_4_crop_{pose_token}_output_d.json"
            return try_load(model_dir / filename)
        if modality == "d" and crop and pose != "Frontal":
            filename = f"dataset_4_crop_{pose_token}_output_d.json"
            return try_load(model_dir / filename)
        if pose == "Frontal":
            filename = f"dataset_4_{crop_key}_output_{modality}.json"
            return try_load(model_dir / filename)
        # non-description or uncrop description use poses pattern
        filename_alt = f"dataset_4_poses_{crop_key}_{pose}_output_{modality}.json"
        return try_load(model_dir / filename_alt)

# Expected missing combinations for internvl2 (no JSON exists)
EXPECTED_MISSING_INTERNVL2 = {
    ("Frontal", True, "s"),      # frontal crop sketch only
    ("Frontal", True, "s_d"),    # frontal crop combined
    ("Frontal", False, "d"),     # frontal uncrop description only
}

def is_expected_missing(model: str, crop: bool, pose: str, modality: str) -> bool:
    return model == "internvl2" and (pose, crop, modality) in EXPECTED_MISSING_INTERNVL2


def compute_accuracy(rows: List[dict]) -> Tuple[int, int, float]:
    correct = 0
    total = 0
    for r in rows:
        ans = r.get("answer")
        pred = r.get("predicted_answer")
        if ans and pred:
            total += 1
            if ans == pred:
                correct += 1
    acc = (correct / total * 100.0) if total else 0.0
    return correct, total, acc


@st.cache_data(show_spinner=False)
def open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def render_trial(entry: dict, crop: bool):
    # Resolve local paths
    sketch_remote = entry.get("sketch_path", "")
    sketch_local = map_remote_to_local(sketch_remote, crop=crop, is_sketch=True)

    option_paths: Dict[str, str] = entry.get("option_paths", {})
    options_local: Dict[str, Optional[Path]] = {
        k: map_remote_to_local(v, crop=crop, is_sketch=False) for k, v in option_paths.items()
    }

    answer = entry.get("answer")
    predicted = entry.get("predicted_answer")

    # Layout
    st.subheader("Query Sketch")
    if sketch_local and sketch_local.exists():
        st.image(open_image(sketch_local), caption=sketch_local.name, width=256)
    else:
        st.warning("Sketch not found locally: " + os.path.basename(sketch_remote))

    st.subheader("Candidates (A–D)")
    cols = st.columns(4)
    letters = ["A", "B", "C", "D"]
    for i, letter in enumerate(letters):
        with cols[i]:
            p = options_local.get(letter)
            cap = f"{letter}"
            if p and p.exists():
                img = open_image(p)
                st.image(img, caption=f"{cap}: {p.name}", use_container_width=True)
            else:
                missing_base = os.path.basename(option_paths.get(letter, ''))
                # Attempt direct pose folder search then similarity fallback
                fb = search_pose_dirs(missing_base, crop=crop) or fallback_search_similar(missing_base, crop=crop)
                if fb and fb.exists():
                    st.image(open_image(fb), caption=f"{cap} (fallback): {fb.name}", use_container_width=True)
                    st.caption(f"Original missing: {missing_base}")
                else:
                    st.warning(f"{cap}: missing {missing_base}")

    # Verdict
    correct = (answer == predicted) if (answer and predicted) else None
    if correct is True:
        st.success(f"Correct: predicted {predicted}")
    elif correct is False:
        st.error(f"Incorrect: predicted {predicted}, correct {answer}")
    else:
        st.info(f"Ground truth: {answer or '-'} | Predicted: {predicted or '-'}")

    with st.expander("Raw JSON entry"):
        st.json(entry)


# ---- UI ----
st.set_page_config(page_title="Sketch+Description Face ID Demo", layout="wide")
st.title("Sketch/Description-Based Face Identification — Demo & Analytics")

st.markdown(
    "This app lets you: (1) explore a single identity (sketch, frontal & pose images, trial result) and (2) view aggregate accuracy analytics across models, poses, crops, and modalities."
)

# ---- Utility for descriptions (placeholder) ----
@st.cache_data(show_spinner=False)
def load_descriptions() -> Dict[str, str]:
    """Load structured descriptions.

    Supports two schemas:
    1. Simple mapping: {"00013": "paragraph ..."}
    2. List of objects: [{"image_name": "00013_...png", "combined_facial": "{ \"Forehead\": ... }"}, ...]

    Returns dict: identity_id -> {"regions": {...}, "paragraph": str}
    """
    candidates = [
        WORKSPACE / "descriptions.json",
        WORKSPACE / "question_jsons" / "descriptions.json",
    ]
    data_raw = None
    for c in candidates:
        if c.exists():
            try:
                data_raw = json.loads(c.read_text(encoding="utf-8"))
                break
            except Exception:
                return {}
    if data_raw is None:
        return {}

    result: Dict[str, Dict[str, object]] = {}

    # Case 1: simple mapping
    if isinstance(data_raw, dict):
        for k, v in data_raw.items():
            identity = k.split("_")[0]  # tolerate keys like 00013_931230
            result[identity] = {
                "regions": {"full": v},
                "paragraph": v,
            }
        return result

    # Case 2: list of objects
    if isinstance(data_raw, list):
        for obj in data_raw:
            if not isinstance(obj, dict):
                continue
            image_name = obj.get("image_name", "")
            identity = image_name.split("_")[0] if image_name else None
            combined_facial = obj.get("combined_facial", "")
            if not identity:
                continue
            regions = {}
            paragraph_parts = []
            if combined_facial:
                try:
                    # combined_facial may itself be a JSON string of region mapping
                    regions = json.loads(combined_facial)
                except Exception:
                    regions = {"raw": combined_facial}
                # Build a readable paragraph
                for key in [
                    "Forehead","Eyes","Eyebrows","Nose","Cheeks","Cheekbones","Jawline","Chin","Mouth & Lips","Ears","Facial Structure","age_gender_ethnicity"
                ]:
                    val = regions.get(key)
                    if val:
                        paragraph_parts.append(f"{key}: {val}")
            paragraph = " " .join(paragraph_parts) if paragraph_parts else combined_facial
            result[identity] = {
                "regions": regions,
                "paragraph": paragraph,
            }
        return result

    return {}

DESCRIPTIONS = load_descriptions()

def get_identity_id_from_frontal(path_str: str) -> Optional[str]:
    base = os.path.basename(path_str)
    if "_" in base:
        return base.split("_")[0]
    # sketch file may be like 00013.jpg
    stem = Path(base).stem
    return stem

def collect_pose_images(identity_id: str, crop: bool) -> List[Path]:
    result = []
    pose_root = DATA_DIRS["poses_crop"] if crop else DATA_DIRS["poses_uncrop"]
    if not pose_root.exists():
        return result
    for pose_folder in sorted(pose_root.iterdir()):
        if not pose_folder.is_dir():
            continue
        # search for prefix match
        for f in pose_folder.glob(f"{identity_id}_*.png"):
            result.append(f)
    return result

@st.cache_data(show_spinner=False)
def list_sketches() -> List[str]:
    if not DATA_DIRS["sketches"].exists():
        return []
    return sorted([p.name for p in DATA_DIRS["sketches"].iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])

def find_trial_for_sketch(rows: List[dict], sketch_name: str) -> Optional[dict]:
    for r in rows:
        sp = r.get("sketch_path", "")
        if sp.lower().endswith(sketch_name.lower()):
            return r
    return None

def find_trial_by_identity(rows: List[dict], identity_id: str) -> Optional[dict]:
    """Fallback: find first trial whose sketch basename starts with identity_id_"""
    for r in rows:
        sp = os.path.basename(r.get("sketch_path", ""))
        if sp.startswith(identity_id + "_"):
            return r
    return None

def find_frontal_image(identity_id: str, crop: bool) -> Optional[Path]:
    """Heuristic frontal image lookup when no trial data is available.

    Searches gallery images directory for files starting with identity_id and containing '_fa'.
    Preference order: *_fa.png then *_fa_a.png then any *_fa*.png.
    """
    gallery_dir = DATA_DIRS["images_crop"] if crop else DATA_DIRS["images_uncrop"]
    if not gallery_dir.exists():
        return None
    # Collect candidates
    primary = sorted(gallery_dir.glob(f"{identity_id}_*_fa.png"))
    alt = sorted(gallery_dir.glob(f"{identity_id}_*_fa_a.png"))
    any_fa = sorted(gallery_dir.glob(f"{identity_id}_*_fa*.png"))
    for pool in (primary, alt, any_fa):
        if pool:
            return pool[0]
    return None

def get_stable_frontal(identity_id: str, crop: bool) -> Optional[Path]:
    """Always return a frontal (non-pose) image for identity.

    Prefers *_fa.png then *_fa_a.png without any '_p_' pose suffix.
    Uses FILE_INDEX for quick lookup.
    """
    basenames = [f for f in FILE_INDEX.keys() if f.startswith(identity_id + "_")]
    pure_frontal = [b for b in basenames if '_fa.png' in b and '_fa_p_' not in b]
    pure_frontal_a = [b for b in basenames if '_fa_a.png' in b and '_fa_p_' not in b]
    ordered = pure_frontal + pure_frontal_a
    for name in ordered:
        paths = FILE_INDEX.get(name, [])
        if paths:
            return _prefer_path(paths, crop=crop)
    # Fallback: derive from pose filenames
    pose_variants = [b for b in basenames if '_fa_p_' in b]
    for pose_name in pose_variants:
        base_part = pose_name.split('_fa_p_')[0]
        for suffix in ['_fa.png', '_fa_a.png']:
            candidate = base_part + suffix
            paths = FILE_INDEX.get(candidate, [])
            if paths:
                return _prefer_path(paths, crop=crop)
    return None

@st.cache_data(show_spinner=False)
def aggregate_accuracy() -> pd.DataFrame:
    records = []
    # Iterate model folders
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for file in model_dir.glob("*.json"):
            fname = file.name
            # Parse modality & pose & crop from filename heuristically
            modality = None
            pose = None
            crop = None
            if "_output_f_" in fname:  # facellm pattern
                modality_part = fname.split("_output_f_")[-1].replace(".json", "")
                modality = modality_part
            elif "_output_" in fname:
                modality_part = fname.split("_output_")[-1].replace(".json", "")
                modality = modality_part
            crop = "crop" if "_crop_" in fname or fname.startswith("dataset_4_crop") else ("uncrop" if "_uncrop_" in fname or fname.startswith("dataset_4_uncrop") else "uncertain")
            # pose token after crop/uncrop segment maybe
            if "_poses_" in fname:
                # e.g., dataset_4_poses_uncrop_0_30_output_f_s_d.json -> pose after uncrop_
                try:
                    parts = fname.split("_poses_")[-1].split("_output")[0].split("_")
                    # parts example: ['uncrop', '0', '30'] or ['uncrop', '0', '30'] with combined '0' or '0', '30'
                    # rebuild pose segment excluding crop key
                    if parts[0] in {"crop", "uncrop"}:
                        pose_parts = parts[1:]
                    else:
                        pose_parts = parts
                    pose = "_".join(pose_parts) if pose_parts else "Frontal"
                    if pose == "0":
                        pose = "Frontal"
                except Exception:
                    pose = "Frontal"
            else:
                pose = "Frontal"
            # Load and compute
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                continue
            correct, total, acc = compute_accuracy(data if isinstance(data, list) else [])
            records.append({
                "model": model,
                "file": fname,
                "modality": modality,
                "pose": pose,
                "crop": crop,
                "correct": correct,
                "total": total,
                "accuracy": acc,
            })
    return pd.DataFrame(records)

ACC_DF = aggregate_accuracy()

tabs = st.tabs(["Single Identity", "Trials Browser", "Analytics"])

# ---- Tab 1: Single Identity ----
with tabs[0]:
    st.subheader("Single Identity Explorer")
    st.markdown("Select a sketch and view its frontal image, pose variants, description (if available), and trial result under chosen conditions.")
    # Sketch selection: either upload or pick existing
    st.markdown("### Sketch Selection")
    sel_col, up_col = st.columns([2, 1])
    uploaded_file = None
    with up_col:
        uploaded_file = st.file_uploader("Upload sketch (.png/.jpg)", type=["png", "jpg"], key="upload_sketch")
    sketch_name: Optional[str] = None
    uploaded_image_obj: Optional[Image.Image] = None
    with sel_col:
        sketch_name = st.selectbox("Or choose from dataset", list_sketches())
    if uploaded_file is not None:
        sketch_name = uploaded_file.name  # override
        try:
            uploaded_image_obj = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Failed to read uploaded image.")
            uploaded_image_obj = None
    st.markdown("### Trial Condition Selection")
    cond_cols = st.columns(4)
    with cond_cols[0]:
        model_single = st.selectbox("Model", ["facellm_38b", "facellm_8b", "internvl2"], index=0, key="single_model")
    with cond_cols[1]:
        modality_single_label = st.selectbox("Modality", ["s", "d", "s_d"], index=2, key="single_modality")
    with cond_cols[2]:
        pose_single = st.selectbox("Pose", ["Frontal", "0_30", "0__45", "15_0", "15_30", "15__30"], index=0, key="single_pose")
    with cond_cols[3]:
        crop_single_variant = st.selectbox("Crop", ["Uncrop", "Crop"], index=0, key="single_crop")
    crop_single = crop_single_variant == "Crop"
    # Load appropriate results
    path_single, rows_single = load_results_file(model_single, crop=crop_single, pose=pose_single, modality=modality_single_label)
    if rows_single is None:
        if is_expected_missing(model_single, crop_single, pose_single, modality_single_label):
            st.info("Expected missing: this internvl2 combination has no results file.")
        else:
            st.warning("Results file not found for current selection.")
        st.stop()
    trial = find_trial_for_sketch(rows_single, sketch_name)  # attempt even if uploaded
    if trial is None and uploaded_file is not None:
        # Try identity-based fallback
        identity_candidate = Path(sketch_name).stem.split("_")[0]
        trial = find_trial_by_identity(rows_single, identity_candidate)
    if trial is None and uploaded_file is None:
        st.warning("No trial entry for this sketch in selected condition.")
        st.stop()
    if trial is None and uploaded_file is not None:
        st.info("Uploaded sketch not found in trials; showing gallery info only.")
    # Identity id first
    if trial:
        answer_letter = trial.get("answer")
        frontal_remote = trial.get("option_paths", {}).get(answer_letter, "")
        identity_id = get_identity_id_from_frontal(frontal_remote) or Path(sketch_name).stem.split("_")[0]
        frontal_basename = os.path.basename(frontal_remote)
        frontal_local = find_local_image(frontal_basename, crop=crop_single)
        # Fallback if mapping failed
        if not frontal_local:
            frontal_local = find_frontal_image(identity_id, crop=crop_single)
            frontal_basename = frontal_local.name if frontal_local else frontal_basename
    else:
        identity_id = Path(sketch_name).stem.split("_")[0]
        frontal_local = find_frontal_image(identity_id, crop=crop_single)
        frontal_basename = frontal_local.name if frontal_local else "(no trial data)"

    # Stable frontal (independent of pose selection)
    stable_frontal = get_stable_frontal(identity_id, crop=crop_single) or frontal_local

    st.markdown("### Query & Frontal")
    gap_cols = st.columns([1,0.15,1])
    with gap_cols[0]:
        if uploaded_image_obj is not None:
            st.image(uploaded_image_obj, caption=f"Sketch: {sketch_name}", width=220)
        else:
            sketch_path_local = DATA_DIRS["sketches"] / sketch_name
            if sketch_path_local.exists():
                st.image(open_image(sketch_path_local), caption=f"Sketch: {sketch_name}", width=220)
            else:
                st.error("Sketch file missing locally.")
    with gap_cols[2]:
        if stable_frontal and stable_frontal.exists():
            st.image(open_image(stable_frontal), caption=f"Frontal: {stable_frontal.name}", width=220)
        else:
            st.warning("Stable frontal image not found.")
    # Description
    st.markdown("### Description")
    desc_entry = DESCRIPTIONS.get(identity_id)
    if not desc_entry:
        st.info("Description not available for this identity.")
    else:
        regions = desc_entry.get("regions", {}) or {}
        display_keys = [
            "Forehead","Eyes","Eyebrows","Nose","Cheeks","Cheekbones","Jawline","Chin","Mouth & Lips","Ears","Facial Structure","age_gender_ethnicity"
        ]
        pretty_names = {"age_gender_ethnicity": "Age/Gender/Ethnicity"}
        lines = []
        for k in display_keys:
            if k in regions and regions[k]:
                name = pretty_names.get(k, k)
                lines.append(f"{name}: {regions[k]}")
        paragraph = "\n".join(lines) if lines else desc_entry.get("paragraph", "")
        st.text(paragraph)
        with st.expander("Raw Regions JSON"):
            st.json(regions)
    # Pose variants
    st.markdown("### Pose Variants (Synthetic)")
    pose_images = collect_pose_images(identity_id, crop=crop_single)
    if not pose_images:
        st.info("No pose images found for identity.")
    else:
        cols_pose = st.columns(min(5, len(pose_images)))
        for i, img_path in enumerate(pose_images[:25]):  # cap display
            with cols_pose[i % len(cols_pose)]:
                st.image(open_image(img_path), caption=img_path.parent.name + ":" + img_path.name, use_container_width=True)
    # Trial candidates
    if trial:
        st.markdown("### Four-Option Trial")
        option_paths_trial: Dict[str, str] = trial.get("option_paths", {})
        trial_cols = st.columns(4)
        for i, letter in enumerate(["A", "B", "C", "D"]):
            with trial_cols[i]:
                remote = option_paths_trial.get(letter, "")
                basename = os.path.basename(remote)
                local_img = find_local_image(basename, crop=crop_single)
                if local_img and local_img.exists():
                    st.image(open_image(local_img), caption=f"{letter} {basename}", use_container_width=True)
                else:
                    st.warning(f"{letter}: missing {basename}")
        # Verdict
        ans = trial.get("answer")
        pred = trial.get("predicted_answer")
        if ans and pred:
            if ans == pred:
                st.success(f"Correct: {pred}")
            else:
                st.error(f"Incorrect: predicted {pred}, answer {ans}")
        else:
            st.info("No prediction available.")
        with st.expander("Raw Trial JSON"):
            st.json(trial)
    else:
        st.info("Uploaded sketch has no associated trial data; only basic display shown.")

# ---- Tab 2: Trials Browser (original functionality) ----
with tabs[1]:
    st.subheader("Trials Browser")
    with st.sidebar:
        st.header("Browser Controls")
        model = st.selectbox("Model", ["facellm_38b", "facellm_8b", "internvl2"], index=0, key="browser_model")
        pose = st.selectbox("Pose", ["Frontal", "0_30", "0__45", "15_0", "15_30", "15__30"], index=0, key="browser_pose")
        crop_variant = st.selectbox("Crop", ["Uncrop", "Crop"], index=0, key="browser_crop")
        crop = crop_variant == "Crop"
        modality_label = st.selectbox("Query Modality", ["Sketch-only (s)", "Description-only (d)", "Combined (s_d)"], index=2, key="browser_modality")
        modality = {"Sketch-only (s)": "s", "Description-only (d)": "d", "Combined (s_d)": "s_d"}[modality_label]
        st.markdown("---")
        only_incorrect = st.checkbox("Show only incorrect trials", value=False, key="browser_incorrect")
        randomize = st.checkbox("Randomize order", value=False, key="browser_random")
    path, rows = load_results_file(model, crop=crop, pose=pose, modality=modality)
    if rows is None:
        if is_expected_missing(model, crop, pose, modality):
            st.info("Expected missing: internvl2 does not include this combination.")
        else:
            st.warning("No result file found for this combination.")
        st.stop()
    correct, total, acc = compute_accuracy(rows)
    st.markdown("### Summary")
    st.write(f"File: `{path.name}` | Model: `{model}` | Pose: `{pose}` | Crop: `{crop_variant}` | Modality: `{modality}`")
    col1, col2, col3 = st.columns(3)
    col1.metric("Correct", f"{correct}")
    col2.metric("Total", f"{total}")
    col3.metric("Accuracy", f"{acc:.2f}%")
    display_rows = rows
    if only_incorrect:
        display_rows = [r for r in rows if r.get("predicted_answer") and r.get("answer") and r["predicted_answer"] != r["answer"]]
    if randomize:
        import random
        random.seed(42)
        random.shuffle(display_rows)
    st.markdown("---")
    st.subheader("Trials")
    idx = st.number_input("Trial index", min_value=0, max_value=max(0, len(display_rows) - 1), value=0, step=1, key="browser_index")
    if not display_rows:
        st.info("No trials to display for the chosen filters.")
    else:
        render_trial(display_rows[int(idx)], crop=crop)

# ---- Tab 3: Analytics ----
with tabs[2]:
    st.subheader("Aggregate Analytics")
    st.markdown("Refine filters and explore interactive charts, KPIs, and heatmaps.")
    if ACC_DF.empty:
        st.warning("No accuracy data could be aggregated.")
        st.stop()
    filt_cols = st.columns(5)
    with filt_cols[0]:
        models_sel = st.multiselect("Models", sorted(ACC_DF.model.unique()), default=sorted(ACC_DF.model.unique()))
    with filt_cols[1]:
        modalities_sel = st.multiselect("Modalities", sorted(ACC_DF.modality.unique()), default=sorted(ACC_DF.modality.unique()))
    with filt_cols[2]:
        crops_sel = st.multiselect("Crop", sorted(ACC_DF.crop.unique()), default=[c for c in ["crop","uncrop"] if c in ACC_DF.crop.unique()])
    with filt_cols[3]:
        poses_sel = st.multiselect("Poses", sorted(ACC_DF.pose.unique()), default=sorted(ACC_DF.pose.unique()))
    with filt_cols[4]:
        show_heatmap = st.checkbox("Show Heatmap", value=True)
    df_f = ACC_DF[(ACC_DF.model.isin(models_sel)) & (ACC_DF.modality.isin(modalities_sel)) & (ACC_DF.crop.isin(crops_sel)) & (ACC_DF.pose.isin(poses_sel))]
    if df_f.empty:
        st.warning("No rows match current filters.")
        st.stop()
    # Ensure numeric types
    df_f = df_f.copy()
    df_f['accuracy'] = pd.to_numeric(df_f['accuracy'], errors='coerce')
    df_f['correct'] = pd.to_numeric(df_f['correct'], errors='coerce')
    df_f['total'] = pd.to_numeric(df_f['total'], errors='coerce')
    # Remove rows with NaN accuracy
    df_f = df_f.dropna(subset=['accuracy'])
    if df_f.empty:
        st.error("Filtered data has no valid accuracy values after type coercion.")
        st.dataframe(ACC_DF.head())
        st.stop()
    # KPIs
    kpi_cols = st.columns(4)
    overall_acc = df_f.accuracy.mean()
    best_row = df_f.loc[df_f.accuracy.idxmax()] if not df_f.empty else None
    with kpi_cols[0]:
        st.metric("Filtered Avg Accuracy", f"{overall_acc:.2f}%")
    with kpi_cols[1]:
        if best_row is not None:
            st.metric("Best Condition", f"{best_row.accuracy:.2f}%", help=f"{best_row.model} | {best_row.modality} | {best_row.crop} | {best_row.pose}")
    with kpi_cols[2]:
        st.metric("Trials Considered", f"{int(df_f.total.sum())}")
    with kpi_cols[3]:
        st.metric("Correct Trials", f"{int(df_f.correct.sum())}")
    st.markdown("### Detailed Table")
    st.dataframe(df_f.sort_values(["model","modality","crop","pose"]))
    # Charts using Altair
    # Altair import guard
    try:
        import altair as alt  # type: ignore
        alt_available = True
    except Exception as e:
        alt_available = False
        st.warning(f"Altair not available ({e}); using fallback charts.")
    st.markdown("### Pose-wise Accuracy (Per Model)")
    pose_order = ["Frontal","0_30","0__45","15_0","15_30","15__30"]
    pose_chart_df = df_f.groupby(["pose","model","modality"], as_index=False)["accuracy"].mean()
    pose_chart_df["pose"] = pd.Categorical(pose_chart_df["pose"], categories=pose_order, ordered=True)
    pose_chart_df = pose_chart_df.sort_values("pose")
    if alt_available and not pose_chart_df.empty:
        models_in_view = sorted(pose_chart_df.model.unique())
        cols_models = st.columns(min(3, len(models_in_view)))
        y_max = max(100, pose_chart_df.accuracy.max()+5)
        for i, m in enumerate(models_in_view[:3]):
            sub = pose_chart_df[pose_chart_df.model == m]
            # Multi-modality colored lines per model
            base = alt.Chart(sub).encode(
                x=alt.X("pose", title="Pose", sort=pose_order),
                y=alt.Y("accuracy", title="Accuracy (%)", scale=alt.Scale(domain=[0, y_max])),
                color=alt.Color("modality:N", title="Modality", scale=alt.Scale(scheme="category10")),
                tooltip=["pose","modality","accuracy"]
            )
            line = base.mark_line(size=3)
            points = base.mark_point(filled=True, size=65, stroke="black", strokeWidth=0.4)
            chart = (line + points).properties(height=280, title=m)
            with cols_models[i]:
                st.altair_chart(chart, use_container_width=True)
        if len(models_in_view) > 3:
            st.info("Only first 3 models shown; refine selection to view others.")
    else:
        # Fallback: combined line chart
        pivot = pose_chart_df.pivot_table(index="pose", columns="model", values="accuracy")
        st.line_chart(pivot)
    st.markdown("### Modality Comparison (Line)")
    modality_chart_df = df_f.groupby(["modality","model"], as_index=False)["accuracy"].mean()
    if alt_available and not modality_chart_df.empty:
        chart_modality = alt.Chart(modality_chart_df).mark_line(point=True).encode(
            x=alt.X("modality", title="Modality"),
            y=alt.Y("accuracy", title="Accuracy (%)"),
            color=alt.Color("model", title="Model"),
            tooltip=["modality","model","accuracy"],
        ).properties(height=250)
        st.altair_chart(chart_modality, use_container_width=True)
    else:
        st.line_chart(modality_chart_df.pivot_table(index="modality", columns="model", values="accuracy"))
    st.markdown("### Model Comparison (Line across Poses)")
    model_chart_df = df_f.groupby(["model","pose"], as_index=False)["accuracy"].mean()
    if alt_available and not model_chart_df.empty:
        chart_model = alt.Chart(model_chart_df).mark_line(point=True).encode(
            x=alt.X("pose", title="Pose"),
            y=alt.Y("accuracy", title="Accuracy (%)"),
            color=alt.Color("model", title="Model"),
            tooltip=["model","pose","accuracy"],
        ).properties(height=250)
        st.altair_chart(chart_model, use_container_width=True)
    else:
        st.line_chart(model_chart_df.pivot_table(index="pose", columns="model", values="accuracy"))
    if show_heatmap and alt_available:
        st.markdown("### Heatmap (Pose vs Modality, Avg Accuracy)")
        heat_df = df_f.groupby(["pose","modality"], as_index=False)["accuracy"].mean()
        chart_heat = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X("modality", title="Modality"),
            y=alt.Y("pose", title="Pose"),
            color=alt.Color("accuracy", scale=alt.Scale(scheme="blues"), title="Acc (%)"),
            tooltip=["pose","modality", alt.Tooltip("accuracy", format=".2f")]
        ).properties(height=320)
        # Overlay text labels
        text_heat = alt.Chart(heat_df).mark_text(fontSize=11, color="black").encode(
            x="modality", y="pose", text=alt.Text("accuracy:Q", format=".1f")
        )
        chart_heat = (chart_heat + text_heat).interactive()
        st.altair_chart(chart_heat, use_container_width=True)
    elif show_heatmap and not alt_available:
        st.info("Heatmap requires Altair; install altair to enable.")
    with st.expander("Raw Aggregated Data"):
        st.dataframe(ACC_DF)
