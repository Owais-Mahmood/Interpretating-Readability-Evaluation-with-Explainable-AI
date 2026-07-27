FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[nlp]"
COPY configs ./configs
COPY scripts ./scripts
ENTRYPOINT ["xai-pipeline"]
CMD ["dry-run", "--config", "configs/base.yaml"]
