#!/usr/bin/env python3
"""Run six ROOT-file exercises with synthetic data and checked reference results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import awkward as ak
import numpy as np
import uproot
from root_io.fixtures import create_example_fixture


def run_lab():
    # A fresh temporary directory prevents overwriting student or analysis files.
    with tempfile.TemporaryDirectory(prefix="root-student-lab-") as directory:
        path = create_example_fixture(Path(directory) / "teaching.root")
        with uproot.open(path, handler=uproot.source.file.MemmapSource,
                         decompression_executor=uproot.TrivialExecutor(),
                         interpretation_executor=uproot.TrivialExecutor()) as f:
            # 1. Inspect the real class rather than assuming every object is a TTree.
            classes = f.classnames(cycle=False)
            assert classes["analysis"] == "TTree"
            assert classes["cutflow"] == "TH1D"
            tree = f["analysis"]
            assert tree.num_entries == 4
            # 2. Read only needed columns; validate parallel collections.
            a = tree.arrays(["event_number", "photon_n", "photon_pt", "photon_eta", "trigP", "mcWeight"], library="ak")
            lengths = ak.num(a.photon_pt, axis=1)
            assert bool(ak.all(lengths == a.photon_n))
            assert bool(ak.all(lengths == ak.num(a.photon_eta, axis=1)))
            # 3. Apply an event mask before accessing the second object.
            selected = a[(lengths >= 2) & a.trigP]
            ids = ak.to_list(selected.event_number)
            second_pt = ak.to_list(selected.photon_pt[:, 1])
            assert ids == [1001, 1002, 1004]
            assert second_pt == [44.0, 38.0, 42.0]
            # 4. Independent weighted-event sum: variance estimate is sum(w**2).
            weights = ak.to_numpy(selected.mcWeight)
            yield_value = float(np.sum(weights))
            variance = float(np.sum(weights**2))
            assert np.isclose(yield_value, 2.90)
            assert np.isclose(variance, 2.8094)
            # 5. An object histogram counts objects, not selected events.
            flat_pt = ak.to_numpy(ak.flatten(selected.photon_pt))
            bins = [0., 40., 60., 80.]
            counts, _ = np.histogram(flat_pt, bins=bins)
            assert counts.tolist() == [2, 3, 2]
            object_weights = ak.to_numpy(ak.flatten(ak.broadcast_arrays(selected.mcWeight, selected.photon_pt)[0]))
            weighted_bins, _ = np.histogram(flat_pt, bins=bins, weights=object_weights)
            assert np.allclose(weighted_bins, [1.88, 2.90, 1.93])
            # 6. These bin contents are declared cumulative cutflow counts.
            # They are a separate teaching example, NOT counts of this four-event tree.
            cutflow = f["cutflow"].values()
            efficiencies = [float(cutflow[1]/cutflow[0]), float(cutflow[2]/cutflow[1])]
            assert np.allclose(efficiencies, [0.8, 0.525])
            result = {"classes": classes, "entries": tree.num_entries,
                      "selected_event_ids": ids, "second_photon_pt_GeV": second_pt,
                      "selected_weighted_yield": yield_value,
                      "selected_weight_variance": variance,
                      "selected_weight_standard_error": float(np.sqrt(variance)),
                      "object_histogram_counts": counts.tolist(),
                      "object_histogram_weighted_contents": weighted_bins.tolist(),
                      "successive_cut_efficiencies": efficiencies}
    return result


def check_pyroot():
    """Optional regression checks for two corrected ROOT API claims."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    with tempfile.TemporaryDirectory(prefix="root-student-pyroot-") as directory:
        path = str(Path(directory) / "ownership.root")
        f = ROOT.TFile(path, "RECREATE")
        h = ROOT.TH1D("owned", "teaching", 3, 0, 3)
        h.Fill(0.5, 2.)
        h.SetDirectory(0)
        f.Close()
        assert h.Integral() == 2.
        tree = ROOT.TTree("vectors", "vectors")
        values = ROOT.std.vector("double")()
        tree.Branch("values", values)
        values.push_back(7.)
        values.push_back(9.)
        tree.Fill()
        reader = ROOT.TTreeReader(tree)
        value = ROOT.TTreeReaderValue("std::vector<double>")(reader, "values")
        assert reader.Next()
        assert list(value.Get()) == [7., 9.]
        return {"ROOT_version": ROOT.gROOT.GetVersion(),
                "detached_histogram_integral": h.Integral(),
                "reader_vector": [7., 9.]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-pyroot", action="store_true", help="Also check histogram ownership and vector reading; requires PyROOT")
    args = parser.parse_args()
    output = run_lab()
    if args.with_pyroot:
        output["pyroot_checks"] = check_pyroot()
    print(json.dumps(output, indent=2))
