# The code is builded with DistributedDataParallel. 
# # Reprodecing the results in the paper should train the model on 2 GPUs.
# # You can also train this model on single GPU and double config.DATA.TRAIN_BATCH in configs.

# For LTCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12340 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_ltcc.yaml --gpu 2,3 --seed 1 --tag res50_rel2-clovar/s1

# For PRCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12340 main.py --dataset prcc --cfg configs/res50_cels_cal_reliability_prcc.yaml --gpu 2,3 --seed 1 --tag res50_rel2-clovar/s1

# For DeepChange dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12341 main.py --dataset deepchange --cfg configs/res50_cels_cal_reliability_deepchange.yaml --gpu 0,1 --seed 1 --tag res50_rel2-clovar/s1