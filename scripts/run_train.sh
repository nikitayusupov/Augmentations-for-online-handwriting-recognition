CONFIG_NAME=${1:-no_lm_no_aug_lower_case}    

# WANDB_MODE=disabled \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTHONPATH=${PYTHONPATH}:$(pwd) \
python diplom/train.py --config-name ${CONFIG_NAME}