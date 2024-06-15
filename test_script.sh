# # For testing 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.1_0.5/s1/best_model.pth.tar --tag res50_rel2-clovar_0.1_0.5/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.9_0.5/s1/best_model.pth.tar --tag res50_rel2-clovar_0.9_0.5/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.5_0.1/s1/best_model.pth.tar --tag res50_rel2-clovar_0.5_0.1/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.9_0.1/s1/best_model.pth.tar --tag res50_rel2-clovar_0.9_0.1/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.3_0.5/s1/best_model.pth.tar --tag res50_rel2-clovar_0.3_0.5/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.7_0.5/s1/best_model.pth.tar --tag res50_rel2-clovar_0.7_0.5/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.5_0.3/s1/best_model.pth.tar --tag res50_rel2-clovar_0.5_0.3/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.5_0.7/s1/best_model.pth.tar --tag res50_rel2-clovar_0.5_0.7/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar_0.5_0.9/s1/best_model.pth.tar --tag res50_rel2-clovar_0.5_0.9/s1

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset ltcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/ltcc/res50_rel2-clovar/s1/best_model.pth.tar --tag res50_rel2-clovar/s1
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12351 main.py --dataset prcc --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 2,3 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/prcc/res50_rel2_clovar/s1/best_model.pth.tar --tag res50_rel2_clovar/s1/wibfw_wosc

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 main.py --dataset deepchange --cfg configs/res50_cels_cal_reliability_test.yaml --gpu 0,1 --seed 1 --eval --resume /data/zhaojiahe/results/ccreid/logs/deepchange/res50_rel2-clovar-pixelaug/s1/best_model.pth.tar --tag res50_rel2-clovar-pixelaug/s1
