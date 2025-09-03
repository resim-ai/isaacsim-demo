FROM nvcr.io/nvidia/isaac-sim:5.0.0@sha256:499e1fd7a3ffc68c6c6fff466d4a6a6dc9ab1573e0a28dbd5f100501ea64e410

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

# Load shader cache
RUN --mount=type=bind,source=./build/shadercache.tar.gz,target=/shadercache.tar.gz \
    tar -xzf /shadercache.tar.gz -C /