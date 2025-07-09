FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

# Load shader cache
RUN --mount=type=bind,source=./build/isaac-cache.tar.gz,target=/isaac-cache.tar.gz \
    tar -xzf /isaac-cache.tar.gz -C /