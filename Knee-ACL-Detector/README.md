# Knee MRI ACL tear check

This local research app takes a sagittal knee MRI series and reports **Possible ACL tear** or **No ACL tear flagged**. A trained classifier uses the pixel data. The original labels, filenames and radiologist-drawn regions are excluded from its inputs.

## Run on Windows

1. Install Python 3.11 or 3.12 if needed. Extract the entire ZIP into a folder.
2. Double-click **Start-Windows.bat**. The first run installs Python packages and opens a local browser tab. If needed, open the localhost URL printed in the terminal. Closing the terminal stops the app.
3. Select **Try an included scan**, pick an example, and click **Analyze scan**. Expand the recorded finding to compare. The two examples are the first held-out series in each binary label category, chosen independently of prediction correctness.

To launch manually from this folder on Windows, macOS or Linux, use a virtual environment and run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 64 --browser.gatherUsageStats false
```

## Use a scan

Under **My scan**, select an original KneeMRI research volume (`.pck`) or a ZIP containing a single, complete sagittal proton-density fat-suppressed knee series in standard DICOM format. Files are processed in memory by the local Python server. The app performs local inference without an external model API.

For DICOM, use 8–128 uncompressed, classic, single-frame grayscale slices. The program checks series membership, orientation, slice spacing and image dimensions, orders slices by position, and orients the images consistently. The user selecting the series must verify the knee anatomy and acquisition sequence. The system does not automatically recognize every type of hospital MRI. A PDF, phone photo, single screenshot or full multiseries study is unsupported.

## Training and evaluation

The source is the **Clinical Hospital Centre Rijeka KneeMRI dataset** distributed on Kaggle. In the download used here, 736 labeled volumes were present. Another 181 rows had no corresponding scan. Healthy ACL was mapped to the negative class, and the dataset's partial-injury and complete-rupture labels to the positive class.

The CPU baseline uses whole-scan grayscale intensity and gradient-direction summaries from the central area of ten sagittal slice positions, followed by standardized logistic regression. Slice order reversal produces the same features. Four-fold cross-validation within the training group selects regularization by ROC-AUC and a decision threshold by the largest difference between sensitivity and false-positive rate. The final holdout was reserved before either choice. Splits are grouped by **exam ID**. Patient identity across separate exams could not be verified.

| Evaluation item | Result |
| --- | --- |
| Training series | 588 |
| Held-out series | 148 |
| Labeled injuries caught | 20 / 38 |
| Labeled injuries missed | 18 |
| False alarms among healthy ACL series | 29 / 110 |
| Sensitivity | 52.6% |
| Specificity | 73.6% |
| ROC-AUC | 0.706 |

The held-out scans come from the same research collection. Diagnostic performance on independent hospital uploads has not been established. A clinician should review personal scans. The app cannot confirm or rule out a tear.

To reproduce training, download the original ZIP from the Kaggle listing below, install `scikit-learn>=1.5,<2` in addition to the app requirements, and run:

```sh
python train.py path/to/kneemridataset.zip
```

The training script writes weights, aggregate metrics and held-out predictions. The included model is ready to use.

## Data credit

I. Štajduhar, M. Mamula, D. Miletić and G. Ünal. *Semi-automated detection of anterior cruciate ligament injury from MRI*. Computer Methods and Programs in Biomedicine 140, 151–164 (2017). https://doi.org/10.1016/j.cmpb.2016.12.006

The two bundled research volumes are unchanged copies from Clinical Hospital Centre Rijeka, Croatia. The source data are licensed under CC BY-NC-ND 4.0. This educational prototype includes no clinical-use validation.

- [Data on Kaggle](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset)
- [Original dataset and credit](https://zenodo.org/records/14789903)
- [Source data license](https://creativecommons.org/licenses/by-nc-nd/4.0/)
