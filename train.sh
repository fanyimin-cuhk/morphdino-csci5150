/mnt/data/aim/liyaxuan/.conda/envs/treedino/bin/torchrun --nproc_per_node=2 dinov2/train/train.py \
--config-file dinov2/configs/train/test.yaml \
--output-dir /mnt/data/aim/liyaxuan/projects/project2/test/ \
train.dataset_path=NeuronMorpho:split=TRAIN:root=/mnt/data/aim/liyaxuan/projects/project2/sample_predata:extra=/mnt/data/aim/liyaxuan/projects/project2/sample_predata


