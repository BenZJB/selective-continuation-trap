# Dataset notes used by this project

The raw archives contain real batch-distillation experiments. The pipeline only uses the `Operation` sensor and actuator trajectories and GC `timeseries` records for the first experiment.

Observed archive structure:

```text
<modality>/Operation/<mixture>/<operating_point>/<experiment>.csv
```

GC aligned time series:

```text
10_GC.../timeseries/<mixture>/<operating_point>/<experiment>.csv
```

A stable run ID is constructed as:

```text
<mixture>__<operating_point>__<experiment>
```

The GC files expose component mole fractions in `V001` and `V002`. The first-pass endpoint is the maximum final available `_mol_V002` mole fraction.

The metadata anomaly labels are retained for diagnostics only and are **not used as model features**, because doing so would risk injecting an annotation unavailable to a live early-stopping policy.
