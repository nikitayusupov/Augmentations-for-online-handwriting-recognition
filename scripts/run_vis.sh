CONFIG_NAME=${1:-resampl_before_norm}    

WANDB_MODE=disabled \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
PYTHONPATH=${PYTHONPATH}:$(pwd) \
python diplom/vis_item.py --config-name ${CONFIG_NAME}