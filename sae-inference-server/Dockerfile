# Build from the repository root. The base already supplies torch, kiji-inspector,
# FastAPI and uvicorn through its vLLM installation; no GPU is used by this server.
FROM 575lab/kiji-inspector:dev
WORKDIR /opt/sae-server
COPY sae-inference-server/app ./app
ENV SAE_DEVICE=cpu
EXPOSE 8001
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
