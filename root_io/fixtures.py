"""Small deterministic-content ROOT fixtures used by tests and examples."""

from __future__ import annotations

from pathlib import Path


def create_example_fixture(path: str | Path) -> Path:
    """Create a tiny diphoton-like ROOT file; return its resolved path."""

    import awkward as ak
    import numpy as np
    import uproot

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(output) as root_file:
        tree = root_file.mktree(
            "analysis",
            {
                "event_number": "int64",
                "photon_n": "int32",
                "photon_pt": "var * float64",
                "photon_eta": "var * float64",
                "trigP": "bool",
                "mcWeight": "float64",
            },
        )
        tree.extend({
            "event_number": np.array([1001, 1002, 1003, 1004], dtype=np.int64),
            "photon_n": np.array([2, 2, 1, 3], dtype=np.int32),
            "photon_pt": ak.Array([[62.0, 44.0], [55.0, 38.0], [48.0], [71.0, 42.0, 19.0]]),
            "photon_eta": ak.Array([[0.2, -0.4], [1.1, -0.8], [0.5], [0.1, -1.2, 2.0]]),
            "trigP": np.array([True, True, False, True]),
            "mcWeight": np.array([1.02, 0.97, 1.10, 0.91], dtype=np.float64),
        })
        root_file["cutflow"] = (
            np.array([100.0, 80.0, 42.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        )
    return output
