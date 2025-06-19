FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

# Load shader cache
RUN --mount=type=bind,source=./build/isaac-cache.tar.gz,target=/isaac-cache.tar.gz \
    tar -xzf /isaac-cache.tar.gz -C /

# Setup entrypoint and deps
COPY builds/isaacsim/entrypoint.sh /
COPY humble_ws/src/isaacsim/scripts/open_isaacsim_stage.py /scripts/

ENTRYPOINT ["/entrypoint.sh"]