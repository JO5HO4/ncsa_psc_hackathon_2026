#!/usr/bin/env python3
"""Build the first manually authored 50-question ROOT/HEP knowledge pilot."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT_FILES = ("ROOT files manual", "https://root.cern/manual/root_files/")
TREES = ("ROOT trees manual", "https://root.cern/manual/trees/")
HIST = ("ROOT histograms manual", "https://root.cern/manual/histograms/")
RDF = ("ROOT RDataFrame manual", "https://root.cern/manual/data_frame/")
UPROOT = ("Uproot getting started guide", "https://uproot.readthedocs.io/en/stable/basic.html")
ATLAS = ("ATLAS Open Data 13 TeV 2025 details", "https://opendata.atlas.cern/docs/data/for_education/13TeV25_details")

# topic, difficulty, question, answer, required facts, source
ITEMS = [
    ("files.purpose", "introductory", "What is a ROOT file used for in high-energy physics?", "A ROOT file persistently stores serialized objects such as histograms and columnar event datasets. HEP analyses use it to retain structured data and analysis products for later reading.", ["ROOT files persist serialized objects", "They can store histograms and columnar datasets"], ROOT_FILES),
    ("files.objects", "introductory", "Does a ROOT file contain only event trees?", "No. A ROOT file can contain many object classes, including trees, histograms, directories, canvases, and user-defined serialized objects.", ["ROOT files are not limited to trees", "Multiple object classes can coexist"], ROOT_FILES),
    ("files.keys", "introductory", "Why should an analyst list a ROOT file's keys before reading event data?", "The key inventory reveals object names and classes, allowing the analyst to locate trees or histograms and avoid assuming a file layout that may be wrong.", ["Keys identify stored objects", "Inventory should precede assumptions about layout"], ROOT_FILES),
    ("files.directories", "introductory", "What role do directories play inside a ROOT file?", "ROOT directories organize persistent objects hierarchically, much like folders. An object may therefore need a path such as `selection/h_mass`, not only a top-level name.", ["Directories provide hierarchy", "Object lookup may require a directory path"], ROOT_FILES),
    ("files.cycles", "intermediate", "What does the cycle suffix in a ROOT key, such as `hist;2`, represent?", "A cycle distinguishes versions of an object written under the same name. Normal retrieval commonly selects the highest cycle, but reproducible inspection should record or intentionally ignore cycles.", ["Cycles distinguish versions with the same name", "Highest-cycle selection can hide older versions"], ROOT_FILES),
    ("files.open_modes", "introductory", "Which ROOT file mode should be used for inspection without modification?", "Use read-only mode, conventionally `READ`. Modes such as `UPDATE` or `RECREATE` can modify or replace content and are inappropriate for a read-only inspection tool.", ["READ is the appropriate inspection mode", "UPDATE and RECREATE can change data"], ROOT_FILES),
    ("files.recreate", "intermediate", "Why is opening an existing ROOT file with `RECREATE` dangerous?", "`RECREATE` creates the target anew and can replace an existing file. It should never be used when the goal is only to inspect preserved analysis data.", ["RECREATE can replace an existing file", "Inspection should be read-only"], ROOT_FILES),
    ("files.security", "intermediate", "Why should ROOT files from unknown origins be treated cautiously?", "ROOT files may contain more than passive arrays, including serialized objects associated with executable behavior. Use trusted sources and a restricted environment for inspection.", ["ROOT files need not contain only passive data", "Unknown files should be handled in a restricted environment"], ROOT_FILES),
    ("files.provenance", "intermediate", "What metadata is needed to make a ROOT-file training fixture reproducible?", "Record the exact file checksum, immutable dataset revision, source, and expected object path. A filename alone does not uniquely identify unchanged content.", ["A checksum identifies exact content", "An immutable source revision is required"], ROOT_FILES),
    ("files.remote", "intermediate", "Should a training agent be allowed to open arbitrary remote ROOT URLs?", "No. The evaluator should stage approved files locally and expose relative fixture paths. This controls provenance, network dependence, security, and reproducibility.", ["Approved fixtures should be staged locally", "Arbitrary remote access harms reproducibility and safety"], ROOT_FILES),

    ("trees.definition", "introductory", "What is a TTree in ROOT?", "A TTree is a columnar dataset whose entries act like rows and whose branches act like columns, including columns that contain collections or nested structures.", ["Entries correspond to rows", "Branches correspond to columns"], TREES),
    ("trees.entries", "introductory", "What does the number of entries in an event TTree usually indicate?", "It reports the number of stored rows in that tree. In an event-level tree this often corresponds to events, but that interpretation must be confirmed from the dataset documentation.", ["Entries count stored rows", "Event meaning requires dataset context"], TREES),
    ("trees.branches", "introductory", "What is a branch in a TTree?", "A branch is a column-like component of a tree that stores one field or structured group across entries and can often be read independently of unrelated branches.", ["A branch is column-like", "Branches support selective reading"], TREES),
    ("trees.leaves", "intermediate", "How are TTree leaves related to branches?", "A branch organizes stored data and may contain one or more leaves describing primitive values. Modern object branches can be more complex, so branch and leaf should not be assumed to be synonyms.", ["A branch may contain leaves", "Branch and leaf are not always identical concepts"], TREES),
    ("trees.columnar", "introductory", "Why is columnar storage useful for HEP analysis?", "An analysis often needs only a subset of variables from many events. Columnar storage lets it read selected branches instead of materializing every field, reducing I/O and memory use.", ["Analyses can select needed columns", "Selective reading reduces I/O"], TREES),
    ("trees.collections", "intermediate", "Why can a TTree branch be jagged rather than rectangular?", "The number of reconstructed objects varies by event. A jet or photon branch can therefore contain a different-length collection for each entry.", ["Object multiplicity varies by event", "Jagged branches have variable inner lengths"], TREES),
    ("trees.baskets", "advanced", "What purpose do baskets serve in TTree storage?", "Baskets hold batches of serialized branch data. Reading data in suitably sized ranges can reuse this organization more efficiently than many tiny, scattered reads.", ["Baskets group serialized branch data", "Read patterns affect I/O efficiency"], TREES),
    ("trees.clusters", "advanced", "What is a TTree cluster intended to improve?", "Clusters group corresponding basket ranges across branches, supporting more efficient prefetching and chunked processing over entry ranges.", ["Clusters group entry ranges", "They support prefetching and chunking"], TREES),
    ("trees.tchain", "intermediate", "When is a TChain useful in a ROOT analysis?", "A TChain presents compatible TTrees from multiple files as one logical sequence of entries. Branch compatibility and file provenance still need validation.", ["TChain combines compatible trees logically", "Compatibility must be checked"], TREES),
    ("trees.rntuple", "intermediate", "Is RNTuple simply another name for TTree?", "No. RNTuple is a newer ROOT columnar storage system with a different design and interfaces. Inspection code should identify the object class instead of assuming every columnar object is a TTree.", ["RNTuple and TTree are distinct", "Tools should inspect object classes"], TREES),

    ("histograms.definition", "introductory", "What does a one-dimensional ROOT histogram represent?", "It summarizes a numerical distribution by dividing an axis into bins and accumulating a content value for each bin.", ["A histogram divides an axis into bins", "Each bin has accumulated content"], HIST),
    ("histograms.edges", "introductory", "Why does a histogram with N regular bins have N+1 finite bin edges?", "Each bin is an interval between a lower and upper boundary. N adjacent intervals therefore require N+1 boundary values.", ["Bins are intervals between edges", "N adjacent bins require N+1 edges"], HIST),
    ("histograms.flow", "introductory", "What are histogram underflow and overflow bins?", "Underflow collects values below the regular axis range, while overflow collects values at or above the upper range according to ROOT's binning rules. They are separate from regular bins.", ["Underflow is below the regular range", "Overflow is beyond the regular upper range"], HIST),
    ("histograms.integral", "intermediate", "Why might a visible-bin histogram integral differ from the total processed weight?", "The default integral may exclude underflow and overflow, and selections may reject events. Weighted entries also mean the sum of bin contents need not equal a raw event count.", ["Flow bins may be excluded", "Weighted content is not necessarily an event count"], HIST),
    ("histograms.weights", "intermediate", "Can a histogram bin content be interpreted automatically as an event count?", "Not always. With weighted filling, bin content is a sum of weights and may be fractional or negative, so the filling convention must be known.", ["Weighted bin content is a sum of weights", "It may be fractional or negative"], HIST),
    ("histograms.sumw2", "intermediate", "What information does Sumw2 preserve for a weighted histogram?", "It stores the sum of squared weights per bin, which ROOT uses to calculate appropriate statistical uncertainties for weighted contents.", ["Sumw2 stores squared-weight sums", "It supports weighted statistical uncertainties"], HIST),
    ("histograms.errors", "intermediate", "Why should a dataset include histogram uncertainties as well as bin contents?", "Two histograms can have similar contents but very different statistical precision. Errors or variances are needed for quantitative comparisons and fits.", ["Bin contents do not encode precision alone", "Errors or variances matter for fits"], HIST),
    ("histograms.normalization", "intermediate", "What must be checked before normalizing a histogram to unit area?", "Decide whether flow bins and negative weights are included, verify the chosen integral is nonzero, and preserve the original normalization or scaling metadata.", ["The normalization integral must be defined", "Zero integrals and negative weights require care"], HIST),
    ("histograms.dimensions", "introductory", "How does a TH2 differ conceptually from a TH1?", "A TH1 bins one variable along one axis, whereas a TH2 bins pairs of variables across two axes and stores content in two-dimensional cells.", ["TH1 represents one axis", "TH2 represents two axes"], HIST),
    ("histograms.profiles", "advanced", "Why should a TProfile not be described as an ordinary count histogram?", "A profile stores a mean dependent value for each bin of an independent variable, along with profile-specific uncertainty information; its content does not simply count entries.", ["A profile stores conditional means", "Profile content is not a simple count"], HIST),

    ("dataframe.workflow", "introductory", "What are the main stages of an RDataFrame analysis?", "Construct a dataframe from a data source, add transformations such as filters or defined columns, and book actions that produce aggregated results or snapshots.", ["Construction selects the data source", "Transformations precede result actions"], RDF),
    ("dataframe.lazy", "intermediate", "What does lazy execution mean in RDataFrame?", "Transformations and most actions describe a computation graph without immediately looping over events. Execution begins when a result is requested.", ["Operations first build a computation graph", "A requested result triggers execution"], RDF),
    ("dataframe.filter", "introductory", "What does RDataFrame Filter do?", "Filter keeps rows satisfying a boolean condition and passes only those rows to downstream transformations and actions.", ["Filter applies a boolean condition", "Rejected rows do not reach downstream operations"], RDF),
    ("dataframe.define", "introductory", "What does RDataFrame Define do?", "Define creates a new column from existing columns or other inputs without modifying the original stored dataset unless a later output action writes it.", ["Define creates a derived column", "It does not by itself rewrite the source file"], RDF),
    ("dataframe.actions", "intermediate", "Why is it useful to book several RDataFrame actions before reading their results?", "Lazy actions can share an event loop through the same computation graph, avoiding separate full passes when the results are triggered together.", ["Lazy actions can share an event loop", "This can avoid repeated full scans"], RDF),
    ("dataframe.snapshot", "intermediate", "What is RDataFrame Snapshot used for?", "Snapshot writes selected columns from the dataframe computation to a new columnar dataset, commonly a new ROOT file and tree.", ["Snapshot writes a new dataset", "Columns can be selected for output"], RDF),
    ("uproot.role", "introductory", "What is Uproot used for in a Python HEP workflow?", "Uproot reads and writes ROOT files using Python array libraries without requiring the ROOT C++ runtime for supported objects.", ["Uproot provides Python ROOT I/O", "It integrates with array libraries"], UPROOT),
    ("uproot.arrays", "introductory", "What is the difference between Uproot's `array` and `arrays` operations?", "A branch `array` call reads one branch, while a tree `arrays` call reads a selected group of branches into an array collection.", ["array reads one branch", "arrays reads a branch group"], UPROOT),
    ("uproot.libraries", "intermediate", "When should Awkward Array be preferred over NumPy for ROOT data?", "Awkward Array naturally represents variable-length and nested per-event collections. Plain NumPy is most convenient for regular rectangular data.", ["Awkward handles variable-length nested data", "NumPy is suited to regular arrays"], UPROOT),
    ("uproot.chunking", "intermediate", "Why should a large ROOT tree be processed in entry chunks?", "Chunking bounds memory use and allows an analysis to scale beyond files that fit in memory while retaining column selection.", ["Chunking bounds memory use", "It supports datasets larger than memory"], UPROOT),

    ("hep.units", "introductory", "Can the unit of a branch such as `photon_pt` be determined from its name alone?", "No. The name suggests a physical quantity but does not establish units or calibration. Use the dataset documentation or explicit metadata before stating units.", ["Branch names do not prove units", "Documentation or metadata is required"], ATLAS),
    ("hep.multiplicity", "introductory", "How should a branch named `jet_n` be checked against jagged jet branches?", "Compare `jet_n` with the per-entry lengths of jet collections such as `jet_pt`. Disagreement can reveal a schema misunderstanding, truncation, or malformed data.", ["Compare multiplicity with collection lengths", "Disagreement requires investigation"], ATLAS),
    ("hep.kinematics", "intermediate", "Why should an agent avoid calculating invariant mass before inspecting branch types and units?", "The calculation requires compatible numeric collections, correct object pairing, and consistent units. Skipping schema and metadata checks can produce plausible but meaningless values.", ["Invariant mass needs compatible inputs", "Units and pairing must be established"], ATLAS),
    ("hep.data_mc", "introductory", "Why must an analyst distinguish collision data from simulated events before interpreting weights?", "Simulation commonly carries generator and correction weights, while collision data follows different normalization conventions. Applying MC weights to data would be incorrect.", ["Simulation and data use different weighting conventions", "MC weights should not be applied blindly to data"], ATLAS),
    ("hep.weights", "intermediate", "Why can a Monte Carlo event weight be negative?", "Some higher-order event generators produce signed weights as part of their estimation procedure. Negative weights are not automatically corrupt data and must be retained according to the sample prescription.", ["Some generators produce signed weights", "Negative weights should not be discarded automatically"], ATLAS),
    ("hep.scale_factors", "intermediate", "What should be verified before multiplying several scale-factor branches into an event weight?", "Confirm each factor's definition, applicability, systematic variation, and whether it is already included elsewhere. Otherwise corrections can be omitted or double counted.", ["Scale-factor definitions and applicability must be checked", "Double counting is a risk"], ATLAS),
    ("hep.triggers", "intermediate", "What does a boolean trigger branch establish by itself?", "It establishes the stored pass/fail value for that branch. It does not by itself document the trigger menu, prescale, efficiency, or suitability for a physics selection.", ["A trigger branch stores a decision", "Menu and efficiency require additional documentation"], ATLAS),
    ("hep.cutflow", "introductory", "What is the purpose of a cut-flow histogram in a HEP analysis?", "A cut flow records how the event yield changes through an ordered sequence of selections, helping validate selection logic and locate unexpected losses.", ["Cut flows track yields across selections", "They help diagnose event losses"], ATLAS),
    ("hep.validation", "intermediate", "What basic checks should be made before using a newly received event ROOT file?", "Verify its checksum and provenance, inventory its objects, confirm the expected tree and entry count, inspect branch names and types, and test a bounded sample for readable values.", ["Check provenance and checksum", "Inspect schema before bulk reading"], ATLAS),
    ("hep.reproducibility", "intermediate", "Why should train and test questions from near-identical ROOT files stay in the same split group?", "Otherwise the model may memorize the shared schema or sample-specific facts and appear to generalize on a test set that is effectively duplicated from training.", ["Near-identical fixtures can leak information", "Fixture grouping makes evaluation more credible"], ATLAS),
]


def main() -> None:
    if len(ITEMS) != 50:
        raise RuntimeError(f"Expected 50 pilot items, found {len(ITEMS)}")
    output = Path(__file__).with_name("question_bank") / "pilot.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked_groups = sorted(
        (item[0] for item in ITEMS),
        key=lambda topic: hashlib.sha256(topic.encode()).hexdigest(),
    )
    split_by_group = {
        topic: "train" if rank < 40 else "validation" if rank < 45 else "test"
        for rank, topic in enumerate(ranked_groups)
    }
    records = []
    for index, (topic, difficulty, question, answer, facts, source) in enumerate(ITEMS, 1):
        records.append({
            "schema_version": "v0.1",
            "id": f"root_knowledge_{index:03d}",
            "question_kind": "grounded_knowledge",
            "topic": topic,
            "difficulty": difficulty,
            "split": split_by_group[topic],
            "question": question,
            "answer": answer,
            "required_facts": facts,
            "forbidden_claims": [],
            "provenance": [{"title": source[0], "url": source[1]}],
            "training_use": ["sft"],
            "rl_eligible": False,
            "split_group": topic,
            "review_status": "draft",
        })
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    print(f"Wrote {len(records)} questions to {output}")


if __name__ == "__main__":
    main()
