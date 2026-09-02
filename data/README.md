# Data

`reference/` contains links to the two Hugging Face reference datasets. They
are Git submodules, so their Parquet files are not stored in this repository.
Leave them unchanged.

`trex_config/` is where we will build the new TRExFitter config dataset. Each task describes a physics goal, a starting `.config` file, and the expected fix.

The first starting config is `trex_config/fixtures/hyy/hyy.config`. It is a working H→γγ TRExFitter config and should be the base for our first example tasks.

`harbor/` is reserved for Harbor data when it arrives.

Write new tasks as JSON records using `trex_config/schema/`. Do not create training Parquet files by hand; the training code will create them later.
