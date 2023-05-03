Implementation of brain MRI tumor segmentation using denoising diffusion GANs.

How to run

- Create conda environment using environment.yml file
- Start training using
python train_ddgan.py --num_epoch 100 --num_process_per_node 2 --num_proc_node 2 --data_dir swin_unitr --json_list swin_unitr/brats21_folds.json
- Run testing using
python test_ddgan.py --data_dir swin_unitr --json_list swin_unitr/brats21_folds.json
