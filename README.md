# Debiasing Through Circuits: A Reproducibility Study in Mechanistic Interpretability


## Notebooks Locations and Usage

When working with notebooks make sure to set the working path as the one for your source directory.

### Experiment 1

- `src/org_paper/reproduction.ipynb` - notebook used to reproduce origial experiments
  
### Experiment 2

- `src/activation_patching/activation_patching_script.py --df_path data/circuit_identification_data/final_toxicity_prompts_0.csv --batch_size 16 --dir_name src/activation_patching/final_toxicity_data_0 --wiki_names_path data/wiki_last_name_master.csv --calculate_patches --image_output_path src/activation_patching/activation_patching.png`  - runs activation patching using authors' method

- circuit identification with EAP/EAP-IG 
  - `python ./scripts/get_circuit.py --config_path ../configs/llama3_config_toxicity.py` runs circuit discovery for the toxicity task.
  -  `python ./scripts/get_circuit.py --config_path ../configs/llama3_config_adv_bias.py` runs circuit discovery for the name-bias task.
  - `work/visualize_circuits/visualize.ipynb` converts the csv files containing the EAP scores and the saved computation graphs in json into visualizations for the paper. It also performs the egde pruning for generating the graphs used in Experiment 5 for bias mitigation. Necessary files are self-contained in the `inputs` and `outputs` subdirectories.


### Experiment 3

- `src/org_paper/grad_experiment_acro.ipynb` - notebook used to generate adversarial samples for authors' task
- `src/adv_sample/sample_generation.ipynb` - notebook used to generate adversarial samples for our task

### Experiment 4

- `src/vul_detect/vul_heads_detection.ipynb` - notebook for creating results of vulnerable heads using authors' method 

### Experiment 5 

- `src/dataset_generation/circuit_adv_data.ipynb` - notebook for creating dataset for bias circuit from adversarial samples

- `src/bias_scale_results.ipynb` - notebook about results of Experiment 5, scaling down the edges to mitigate the bias

### Others

- `src/plots.ipynb` - notebook for additional plots for the paper

