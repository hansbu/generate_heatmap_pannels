# generate_heatmap_pannels

This git repo is to generate 4-image panel as the one in example.png.

There are different branch for different tumor type.

- 2_classes: for tumor type with binary classification, e.g, BRCA, PAAD
- 3_classes_prad: for 3-way classification, especially for PRAD.
- 6_classes_luad: for 6-way classification, especially for LUAD.

The run instructions are the same for all branches.

## Usage
python main.py N

where N can be -1, 0, 1, or any positive integer.
- 0/1: not using parallel processing
- any number larger than 1, using N cores in parallel processing, limited to the available cores in the system.
- -1: use all available cores in parallel processing, left 2 cores for others
