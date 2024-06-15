# For LTCC 
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar/s1/best_model.pth.tar --tag res50_rel2-clovar/s1

# For PRCC
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset prcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 2,3 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/prcc/res50_rel2_clovar/s1/best_model.pth.tar --tag res50_rel2_clovar/s1/

# For DeepChange
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 main.py --dataset deepchange --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/deepchange/res50_rel2-clovar/s1/best_model.pth.tar --tag res50_rel2-clovar/s1
