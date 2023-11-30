/mnt/data/aim/liyaxuan/.conda/envs/treedino/bin/torchrun --nproc_per_node=2 cell_mtype_getvec.py \
--config-file dinov2/configs/train/test.yaml \
--output-dir /mnt/data/aim/liyaxuan/projects/project2/val/ \
train.dataset_path=NeuronMorpho:split=TRAIN:root=/mnt/data/aim/liyaxuan/projects/project2/sample_predata:extra=/mnt/data/aim/liyaxuan/projects/project2/sample_predata