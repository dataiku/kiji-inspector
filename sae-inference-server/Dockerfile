# Build from the repository root. The base already supplies torch, kiji-inspector,
# FastAPI and uvicorn through its vLLM installation; no GPU is used by this server.
FROM 575lab/kiji-inspector:dev
# Use the checked-out package alongside the app; the published base can contain
# an older SAE API (for example, without normalize_input).
COPY src/kiji_inspector /opt/kiji-src/kiji_inspector
ENV PYTHONPATH=/opt/kiji-src
WORKDIR /opt/sae-server
COPY sae-inference-server/app ./app
ENV SAE_DEVICE=cpu
EXPOSE 8001
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
