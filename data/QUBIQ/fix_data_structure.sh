set -e
mkdir -p training_data_v3_QC/pancreas/Training training_data_v3_QC/pancreatic-lesion/Training
for d in training_data_v3_QC/pancreas/*; do [ -d "$d" ] || continue; base=$(basename "$d"); if [ "$base" != "Training" ]; then mv "$d" training_data_v3_QC/pancreas/Training/; fi; done
for d in training_data_v3_QC/pancreatic-lesion/*; do [ -d "$d" ] || continue; base=$(basename "$d"); if [ "$base" != "Training" ]; then mv "$d" training_data_v3_QC/pancreatic-lesion/Training/; fi; done
echo "Move complete"