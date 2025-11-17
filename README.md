## Builds

Build using 
```
./rebuild.sh
```

Run on a system with a GPU:
```
docker compose -f builds/docker-compose.local.yml up --abort-on-container-exit
```

## Shader Cache

See `builds/isaacsim/shader-cache-images` for more details.