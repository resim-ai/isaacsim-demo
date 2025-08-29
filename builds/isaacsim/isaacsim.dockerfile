FROM nvcr.io/nvidia/isaac-sim:5.0.0

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Setup entrypoint and deps
COPY builds/isaacsim/entrypoint.sh /
COPY humble_ws/src/isaacsim/scripts/open_isaacsim_stage.py /scripts/

ENTRYPOINT ["/entrypoint.sh"]