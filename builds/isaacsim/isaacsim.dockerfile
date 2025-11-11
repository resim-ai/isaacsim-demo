FROM nvcr.io/nvidia/isaac-sim:5.0.0

# Setup entrypoint and deps
COPY builds/isaacsim/entrypoint.sh /
COPY humble_ws/src/isaacsim/scripts/open_isaacsim_stage.py /scripts/

ENTRYPOINT ["/entrypoint.sh"]