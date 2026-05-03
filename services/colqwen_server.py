"""ColQwen2 multi-vector encoding service (Litserve).

Start:
    python -m services.colqwen_server [--port 8000]

Endpoints (Litserve auto-generates /predict and /health):
    POST /predict  {"action":"encode_query"|"encode_pages", ...} -> {"embedding": [...]}
    GET  /health   -> {"status":"ok"}

Architecture:
    Chainlit worker (CPU) --HTTP--> ColQwen Service (GPU, single instance)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import litserve as ls
from loguru import logger

from ingestion.model_loader import encode_pages, encode_query, load_model_and_processor


class ColQwenLitAPI(ls.LitAPI):
    def setup(self, device):
        logger.info(f"Litserve assigned device: {device}")
        self.model, self.processor = load_model_and_processor()

    def decode_request(self, request):
        action = request.get("action", "")
        if action == "encode_query":
            return ("encode_query", request["query"])
        elif action == "encode_pages":
            return ("encode_pages", [Path(p) for p in request["image_paths"]])
        raise ValueError(f"Unknown action: {action}")

    def predict(self, inputs):
        action, data = inputs
        if action == "encode_query":
            emb = encode_query(self.model, self.processor, data)
            return {"embedding": emb.tolist()}
        elif action == "encode_pages":
            emb = encode_pages(self.model, self.processor, data, batch_size=1)
            return {"embeddings": emb.tolist()}
        raise ValueError(f"Unknown action: {action}")

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    api = ColQwenLitAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=1,
        workers_per_device=1,
        timeout=False,
    )
    server.run(host=args.host, port=args.port)
