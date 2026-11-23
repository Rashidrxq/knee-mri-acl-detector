import json
import zipfile

import pydicom
import streamlit as st

from dicom_input import read_series
from model import MAX_BYTES, ROOT, dataset_scan, predict

st.set_page_config(page_title="Knee MRI · ACL tear check", page_icon="🦵")
st.title("Check a knee MRI for an ACL tear")
st.write("Choose one scan and run the trained model. Scans stay on this computer when you run the app locally.")
report = json.loads((ROOT/"evaluation.json").read_text())
samples = json.loads((ROOT/"examples"/"index.json").read_text())
mode = st.radio("Scan source", ["My scan", "Try an included scan"], horizontal=True)
scan = None
case = None
confirmed = True
if mode == "Try an included scan":
    case = samples[st.selectbox("Example", list(samples))]
    scan = (ROOT/"examples"/case["file"]).read_bytes()
else:
    file = st.file_uploader("Select a knee MRI", type=["zip","pck"],
        help="Choose one sagittal DICOM series as a ZIP, or an original KneeMRI dataset .pck volume.")
    if file is not None and file.size > MAX_BYTES:
        st.error("The file is too large. Choose one series under 64 MiB.")
    elif file is not None:
        scan = file.getvalue()
    is_dicom = file is not None and file.name.lower().endswith(".zip")
    if is_dicom:
        confirmed = st.checkbox("This ZIP contains one complete sagittal proton-density fat-suppressed knee MRI series.")
        st.caption("Supported hospital export: one uncompressed single-frame DICOM series, standard sagittal orientation. The protocol must be verified by the person selecting the series.")
    else:
        st.caption("Accepted files: a ZIP of DICOM slices or a KneeMRI .pck research volume.")
if st.button("Analyze scan", type="primary", disabled=scan is None or not confirmed):
    try:
        with st.spinner("Analyzing the MRI…"):
            volume = read_series(scan) if mode == "My scan" and is_dicom else dataset_scan(scan)
            flag = predict(volume)
    except (ValueError, KeyError, AttributeError, TypeError, OSError, IndexError, RuntimeError,
            zipfile.BadZipFile, pydicom.errors.InvalidDicomError, NotImplementedError):
        st.error("Unable to analyze this file. Check the format, series selection, and scan orientation.")
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
