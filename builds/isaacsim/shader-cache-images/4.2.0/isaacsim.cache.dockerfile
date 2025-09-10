FROM nvcr.io/nvidia/isaac-sim:4.2.0@sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

# Load shader cache
RUN --mount=type=bind,source=./build/shadercache.tar.gz,target=/shadercache.tar.gz \
    tar -xzf /shadercache.tar.gz -C /