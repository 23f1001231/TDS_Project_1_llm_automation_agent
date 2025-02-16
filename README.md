# LLM-based Automation Agent
```
uv pip compile pyproject.toml -o requirements.txt
```

```
podman build -t $container-name .
```

on wsl
```
podman run --network=host -p 8000:8000 -e AIPROXY_TOKEN=$AIPROXY_TOKEN $container-name
```