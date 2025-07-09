FROM 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-shader-cache

# Setup entrypoint and deps
COPY builds/isaacsim/entrypoint.sh /
COPY humble_ws/src/isaacsim/scripts/open_isaacsim_stage.py /scripts/

ENTRYPOINT ["/entrypoint.sh"]