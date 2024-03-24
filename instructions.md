
## **Activate the environment and right path**
```
conda activate gpu
```

### **Training**
```
python /home/projects/ahmad_mbs/project/train.py  \
    --root_dir /home/projects/ahmad_mbs/ \
    --dataset_dir /home/projects/ahmad_mbs/data/ \
    --model_name unet \
    --epochs 2 \
    --batch_size 3 \
    --gpu 0 \
    --experiment road_seg \
    --band_num 4,6,7
```

### **Testing**

```
python /home/projects/ahmad_mbs/project/test.py \
    --dataset_dir /home/projects/ahmad_mbs/data/ \
    --model_name unet \
    --load_model_name unet_ex_Band123_ep_1000_11-Nov-23.hdf5 \
    --experiment road_seg \
    --gpu 0 \
    --band_num 1,2,3
```

