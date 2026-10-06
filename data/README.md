# Experiment data

## Dataset provenance

- **NASA B0005** (the local experiment calls it B5): [NASA PCoE data-set repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), Battery Data Set. The file here is the saved per-cycle capacity sequence extracted for this experiment.
- **CALCE CS2_38**: [University of Maryland CALCE Battery Group data](https://web.calce.umd.edu/batteries/data/). The provider identifies CS2_38 as a cell cycled at constant current, 1C. The file here is the saved capacity sequence used by this experiment.

These files are copied byte-for-byte from the supplied experiment directory. `source_manifest.json` records relative source locations, sizes and SHA-256 hashes. Full upstream measurement archives and the original extraction script are not included. Consult and cite the respective dataset providers; this release does not override their data terms.

## Schemas

All data CSVs are headerless. Each row corresponds to a sequence index. Capacity and decomposition amplitudes are in Ah.

| File | Rows | Columns |
| --- | ---: | --- |
| capacity/B5.csv | 168 | Capacity |
| capacity/CS2_38.csv | 996 | Capacity |
| decompositions/ALAVMDB5.csv | 168 | 8 IMFs, then observed capacity |
| decompositions/ALAVMDCS238.csv | 996 | 12 IMFs, then observed capacity |
| decompositions/VMDB5.csv | 168 | 4 IMFs, then observed capacity |
| decompositions/VMDCS238.csv | 996 | 4 IMFs, then observed capacity |

The last column of every decomposition CSV matches its capacity sequence. Saved decompositions contain actual component amplitudes; they are not normalized presentation curves.

## Prediction alignment

`results/*-B5.csv` has 107 targets, zero-based capacity indices **61–167**. `results/*-38.csv` has 795 targets, indices **201–995**. Headers are `Last_Column_Actual,Summed_Predictions,Difference`. The saved Difference column retains more numerical precision than the rounded prediction field.

The NASA failure threshold is 1.40 Ah and the CALCE threshold is 0.88 Ah. The verification script reports the first saved test target strictly below the threshold; it does not extrapolate beyond these ranges. Sequence indices are preserved; no claim is made that they equal a separately recorded upstream cycle-ID column.
