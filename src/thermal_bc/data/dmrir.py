"""DMR-IR file listing and patient identifiers."""

import os
import re

# DMR-IR exports use several naming schemes; each alternative captures the patient ID:
# "198-1.txt.jpg" / "R198-10.txt.jpg" -> 198, "TFRON_V185_27-8-2013_0.jpg" -> 185, and
# "IR_2015-04-17_0007.jpg" -> 2015-04-17 (acquisition session, the finest ID available).
DMRIR_PATIENT_ID_PATTERN = r"^R?(\d+)-\d+|^R?TFRON_V(\d+)_|^IR_(\d{4}-\d{2}-\d{2})_"


def patient_id_from_path(path, pattern=DMRIR_PATIENT_ID_PATTERN):
    """Extract the patient identifier from a DMR-IR file path using regex `pattern`.

    The first capture group that matched is used as the ID. Raises if the pattern does not
    match, so every image is always assigned to a patient.
    """
    match = re.search(pattern, os.path.basename(path))
    if match is None:
        raise ValueError(
            f"Could not extract a patient ID from '{os.path.basename(path)}' with "
            f"pattern {pattern!r}. Extend DMRIR_PATIENT_ID_PATTERN for new naming schemes."
        )
    return next(g for g in match.groups() if g is not None)


def list_dmrir_images(database_dir):
    """Return (paths, labels); images in `abnormal/` get label 1, images in `normal/` label 0.

    Files are sorted by name so that the order, and therefore the CV folds, is identical on
    every operating system.
    """
    paths, labels = [], []
    for folder, label in (("abnormal", 1), ("normal", 0)):
        folder_path = os.path.join(database_dir, folder)
        files = sorted(os.listdir(folder_path))
        paths += [os.path.join(folder_path, f) for f in files]
        labels += [label] * len(files)
    return paths, labels
