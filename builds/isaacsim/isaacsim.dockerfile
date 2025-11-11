FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Setup entrypoint and deps
COPY builds/isaacsim/entrypoint.sh /
COPY humble_ws/src/isaacsim/scripts/open_isaacsim_stage.py /scripts/

RUN mkdir -p /isaac-sim/shadercache /isaac-sim/.cache

ENTRYPOINT ["/entrypoint.sh"]
