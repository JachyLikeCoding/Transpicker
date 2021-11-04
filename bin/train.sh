python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  main.py \
--dataset_file coco_split \
--coco_path coco_split \
--output_dir my_outputs \
--coco_path coco_split \
--lr 0.0002 \
--num_queries 300 \
--batch_size 2 \
--enc_layers 3 \
--dec_layers 3

# python3 main.py --dataset_file "coco_split" --coco_path "coco_split" --output_dir "outputs/split_outputs_0614" --dec_layers 3 --enc_layers 3
