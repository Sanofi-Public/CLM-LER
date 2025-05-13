#!/bin/bash
set -xe

# Define the list
list="guo_icu guo_los guo_readmission lab_anemia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_thrombocytopenia new_acutemi new_celiac new_hyperlipidemia new_hypertension new_lupus new_pancan"
model_artifact_stem=clm_ler
model_conversions_artifact=train_ehr_clm/clm_ler_conversions:v0
model_percentiles_artifact=train_ehr_clm/clm_ler_percentiles:v0
labelled_data_stem=labelled_data_for_ehrshot_benchmarks
overall_version="v1"
script=src/clm_ler/model_training/train_timeseries_classifier.py
input_data_artifact=ehrshot_benchmarks/${labelled_data_stem}:v0
model_artifact_name=train_ehr_clm/${model_artifact_stem}:v0
training_args=src/clm_ler/config/ehrshot_training_config.yaml
export WANDB_PROJECT=ehrshot_benchmarks


# Convert the list to a stream and iterate over it
echo $list | while read -r item; do
    for task in $item; do
        echo "Processing $i"
            rm -rf tmp_model_data/
            rm -rf model_artifacts/
            rm -rf output/
            rm -rf tmp_*
            output_model_artifact_name=${model_artifact_stem}_${labelled_data_stem}_${task}_${overall_version}
            remote_directory=s3://some/path/trained_model_${output_model_artifact_name}_${overall_version}

            python3 ${script} --input_data_artifact_name ${input_data_artifact} --model_artifact_name ${model_artifact_name} --training_args ${training_args} --output_model_artifact_name ${output_model_artifact_name} --subdataset_name ${task} --tmp_save_directory tmp_model_data/ --output_model_location ${remote_directory} --unit_conversions_artifact_name ${model_conversions_artifact} --percentiles_conversion_artifact_name ${model_percentiles_artifact}
           rm -rf ${output_model_artifact_name}/*
    done
done