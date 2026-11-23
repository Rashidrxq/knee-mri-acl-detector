"""Train and evaluate using the original Kaggle kneeMRIdataset download ZIP."""
import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import MAX_BYTES, ROOT, dataset_scan, features

SEED = 38


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = {Path(i.filename).name: i for i in archive.infolist() if not i.is_dir()}
        with archive.open(entries["metadata.csv"]) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
        missing = [r["volumeFilename"] for r in rows if r["volumeFilename"] not in entries]
        rows = [r for r in rows if r["volumeFilename"] in entries]
        print(f"Found {len(rows)} labeled series, {len(missing)} metadata entries without pixels.", flush=True)
        records = []
        for i, row in enumerate(rows):
            label = int(row["aclDiagnosis"])
            if label not in (0, 1, 2):
                raise ValueError("Unknown ACL label.")
            info = entries[row["volumeFilename"]]
            if info.file_size > MAX_BYTES:
                raise ValueError("Unexpected scan size.")
            vector = features(dataset_scan(archive.read(info)))
            records.append((row["examId"], row["volumeFilename"], label, vector))
            if (i+1) % 100 == 0:
                print(f"Processed {i+1} / {len(rows)} series", flush=True)
    groups, names, diagnosis, vectors = zip(*records)
    return np.array(groups), np.array(names), np.array(diagnosis), np.stack(vectors), missing


def train(archive_path, out):
    out.mkdir(exist_ok=True, parents=True)
    groups, names, diagnosis, x, missing = load_archive(archive_path)
    y = (diagnosis > 0).astype(int)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    train_ids, test_ids = next(outer.split(x, y, groups))
    assert not (set(groups[train_ids]) & set(groups[test_ids]))
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED+1)
                 .split(x[train_ids], y[train_ids], groups[train_ids]))
    def estimator(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
    best_auc = -1
    for c in (.0001, .001, .01, .1):
        scores = np.empty(len(train_ids))
        for fit, dev in folds:
            pipe = estimator(c).fit(x[train_ids[fit]], y[train_ids[fit]])
            scores[dev] = pipe.decision_function(x[train_ids[dev]])
        auc = roc_auc_score(y[train_ids], scores)
        if auc > best_auc:
            best_auc, best_c, best_scores = auc, c, scores
    # Threshold chosen from out-of-fold training predictions, before held-out evaluation.
    fpr, tpr, cuts = roc_curve(y[train_ids], best_scores)
    threshold = float(cuts[np.argmax((tpr - fpr)[1:]) + 1])
    scaler, logistic = estimator(best_c).fit(x[train_ids], y[train_ids]).steps
    scaler, logistic = scaler[1], logistic[1]
    coef = logistic.coef_[0] / scaler.scale_
    intercept = float(logistic.intercept_[0] - coef @ scaler.mean_)
    scores = x[test_ids] @ coef + intercept
    predicted = scores >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y[test_ids], predicted, labels=[0,1]).ravel())
    weights = {"format":1, "coef":coef.tolist(), "intercept":intercept,
               "decision_threshold": threshold}
    report = {"source":"Clinical Hospital Centre Rijeka / KneeMRI via Kaggle", "seed":SEED,
              "unit":"MRI series, grouped by exam ID", "metadata_without_scan":len(missing), "training_series":len(train_ids),
              "training_exams":len(set(groups[train_ids])), "heldout_series":len(test_ids),
              "heldout_exams":len(set(groups[test_ids])), "diagnosis_counts":dict(Counter(map(int, diagnosis))),
              "sensitivity":tp/(tp+fn), "specificity":tn/(tn+fp), "auroc":float(roc_auc_score(y[test_ids],scores)),
              "true_positive":tp, "true_negative":tn, "false_positive":fp, "false_negative":fn,
              "training_oof_auroc":float(best_auc), "C":best_c}
    (out / "weights.json").write_text(json.dumps(weights))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    with (out / "heldout.csv").open("w", newline="") as fh:
        writer=csv.writer(fh)
        writer.writerow(["exam_id", "file", "acl_label", "model_flag"])
        for i, flag in zip(test_ids,predicted):
            writer.writerow([groups[i],names[i],int(y[i]),int(flag)])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT)
    args=parser.parse_args()
    train(args.archive, args.out)
