FROM nvcr.io/nvidia/isaac-sim:5.1.0@sha256:93b0f99635ab126fb5b33298d513c11520f119f0ee60ff8414ccef67ea977829

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

# Load shader cache
RUN --mount=type=bind,source=./build/shadercache.tar.gz,target=/isaac-sim/shadercache.tar.gz \
    tar -xzf /isaac-sim/shadercache.tar.gz -C /isaac-sim