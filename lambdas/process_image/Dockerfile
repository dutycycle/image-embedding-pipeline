FROM public.ecr.aws/lambda/python:3.12

# install system dependencies
RUN microdnf install -y gcc gcc-c++ python3-devel \
    && microdnf clean all

# install python dependencies. use cpu version of torch.
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers \
    Pillow \
    boto3 \
    moto \
    pyarrow \
    pandas \
    pytest

# create cache directories with appropriate permissions. lambda can only write to /tmp
RUN mkdir -p /tmp/transformers_cache /tmp/torch_home /tmp/huggingface \
    && chmod 777 /tmp/transformers_cache /tmp/torch_home /tmp/huggingface

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV TORCH_HOME=/tmp/torch_home
ENV HF_HOME=/tmp/huggingface

COPY process_image.py ${LAMBDA_TASK_ROOT}
COPY tests.py ${LAMBDA_TASK_ROOT}
COPY conftest.py ${LAMBDA_TASK_ROOT}

CMD [ "process_image.handler" ]