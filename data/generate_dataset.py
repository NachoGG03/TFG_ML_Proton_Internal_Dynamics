## --- Dataset Creation ---
# We will create a script that generates the necessary dataset and saves it in NumPy binary files, so we won't have to wait for the LHAPDF library to calculate everything every time we want to use the data.

import numpy as np
import lhapdf
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  #
from utils.dictionaries import pids


def create_dataset(pdf_set_name="NNPDF40_nnlo_as_01180"):
    # 1. Grid Generation
    npoints = int(1e3)  # Number of points to generate
    # x points. Linear grid from 0.1 to 1, logarithmic grid from 1e-5 to 0.1
    xgrid = np.concatenate(
        (
            np.logspace(-5, -1, npoints // 2, endpoint=False),
            np.linspace(0.1, 1, npoints // 2),
        )
    ).astype(np.float32)

    # Energy Scale Selection
    nenergies = int(1e3)
    q2_grid = np.geomspace(5, 1e6, nenergies).astype(
        np.float32
    )  # Geometric spacing from 5 GeV^2 to 1e6 GeV^2

    # 2. PDF Evaluation
    pdf = lhapdf.mkPDF(pdf_set_name)

    # 3. Dataset Construction
    print(f"Generating dataset for {pdf_set_name}...")
    total_points = npoints * nenergies

    inputs = np.zeros((total_points, 2), dtype=np.float32)
    targets = np.zeros((total_points, len(pids)), dtype=np.float32)
    idx = 0
    # Loop over all parton flavors and energy scales
    for i, q2 in enumerate(tqdm(q2_grid, desc="Processing energy scales")):
        q2_val = float(q2)
        for x in xgrid:
            xfx = pdf.xfxQ2(float(x), q2_val)
            inputs[idx, 0] = x
            inputs[idx, 1] = q2
            targets[idx] = [xfx[pid] for pid in pids]
            idx += 1

    # Save the dataset as a numpy file
    if not os.path.exists("data"):
        os.makedirs("data")
    np.save("data/x_q2_inputs_training.npy", inputs)
    np.save("data/pdf_targets_training.npy", targets)
    np.save("data/pids_info_training.npy", np.array(pids))

    print(f"Dataset generated: {len(inputs)} provided points.")


if __name__ == "__main__":
    create_dataset()
