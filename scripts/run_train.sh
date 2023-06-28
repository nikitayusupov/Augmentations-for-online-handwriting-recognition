CONFIG_NAME=${1:-main_config_no_aug}    

# WANDB_MODE=disabled \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTHONPATH=${PYTHONPATH}:$(pwd) \
python diplom/train.py --config-name ${CONFIG_NAME}