set -e

CUDA_VERSION=102
TORCH_VERSION='1.7.1' #'1.8.0'

pip install -r requirements.txt
pip install -U torch==${TORCH_VERSION} # +cu${CUDA_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
pip install -U --force-reinstall torch-scatter \
                torch-spline-conv \
                torch-sparse \
                torch-cluster \
                -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install -U git+https://github.com/rusty1s/pytorch_geometric