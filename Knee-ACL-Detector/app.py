import json

import streamlit as st

from model import MAX_BYTES, ROOT, predict
from scan_input import ScanInputError, load_scan, middle_slice

st.set_page_config(page_title="Knee MRI · ACL tear check", page_icon="🦵")
st.title("Check a knee MRI for an ACL tear")
st.write("Choose one scan and run the trained model. Scans stay on this computer when you run the app locally.")
report = json.loads((ROOT/"evaluation.json").read_text())
samples = json.loads((ROOT/"examples"/"index.json").read_text())
mode = st.radio("Scan source", ["My scan", "Try an included scan"], horizontal=True)
scan_key = None
example_path = None
if mode == "Try an included scan":
    example = samples[st.selectbox("Example", list(samples))]
    example_path = ROOT / "examples" / example["file"]
    scan_key = ("example", example["file"])
else:
    file = st.file_uploader("Select a knee MRI", type=["zip", "pck"],
        help="Upload a single-scan Kaggle ZIP, a KneeMRI .pck, or a ZIP of one supported DICOM series.")
    st.caption("Kaggle downloads can be uploaded directly as a single-scan ZIP.")
    if file is not None:
        if file.size > MAX_BYTES:
            st.error("Choose one scan of 64 MiB or less.")
        else:
            scan_key = ("upload", file.file_id)

# Keep only the currently selected volume in this browser session.
prepared = st.session_state.get("prepared_scan")
if prepared is not None and prepared[0] != scan_key:
    del st.session_state["prepared_scan"]
    prepared = None
if scan_key is not None and prepared is None:
    try:
        with st.spinner("Opening scan…"):
            if example_path is not None:
                scan = load_scan(example_path.read_bytes(), example_path.name)
            else:
                scan = load_scan(file.getvalue(), file.name)
        prepared = (scan_key, scan)
        st.session_state["prepared_scan"] = prepared
    except ScanInputError as exc:
        st.error(str(exc))
    except OSError:
        st.error("Could not open the selected scan file.")

scan = prepared[1] if prepared is not None else None
confirmed = True
case = None
if scan is not None:
    count = len(scan.volume)
    st.image(middle_slice(scan.volume), width=320,
             caption=f"{scan.source} preview · Middle slice {count // 2 + 1} of {count}")
    if scan.source == "DICOM":
        confirmed = st.checkbox(
            "This ZIP contains one complete sagittal proton-density fat-suppressed knee MRI series.",
            key=f"protocol-{scan_key[1]}",
        )
    else:
        case = next((item for item in samples.values() if item["sha256"] == scan.digest), None)

if st.button("Analyze scan", type="primary", disabled=scan is None or not confirmed):
    try:
        with st.spinner("Analyzing the MRI…"):
            flag = predict(scan.volume)
    except ValueError:
        st.error("Cannot analyze this volume. Check that the scan has usable image contrast.")
    else:
        st.subheader("Possible ACL tear" if flag else "No ACL tear flagged")
        missed, detected = report["false_negative"], report["true_positive"]
        alarm, clear = report["false_positive"], report["true_negative"]
        if flag:
            st.write(f"In held-out testing, the model also flagged {alarm} of {alarm+clear} series labeled healthy.")
        else:
            st.write(f"In held-out testing, the model missed {missed} of {missed+detected} labeled injuries. This result cannot rule out an ACL tear.")
        st.caption("Research prototype. A clinician should interpret a personal MRI.")
        if case:
            with st.expander("Compare with the recorded finding"):
                st.write(case["finding"])
with st.expander("How did this model perform?"):
    st.write(f"On {report['heldout_series']} held-out MRI series from the same source, the model caught "
             f"{report['true_positive']} of {report['true_positive']+report['false_negative']} labeled injuries. "
             f"It missed {report['false_negative']} and gave {report['false_positive']} false alarms among "
             f"{report['true_negative']+report['false_positive']} series labeled healthy.")
    st.write("The model was evaluated on KneeMRI research volumes. Performance on hospital DICOM uploads has not been measured.")
