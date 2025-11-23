import json
import os
from typing import Any, Dict
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _make_bedrock_client() -> boto3.client:
    """Create a Bedrock Runtime client using environment variables."""
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)

def invoke_mistral(prompt: str,
                  max_gen_len: int = 1024,
                  temperature: float = 0.2,
                  top_p: float = 0.9) -> str:
    """
    Call the Mistral‑7B‑Instruct model on Amazon Bedrock.

    Returns the generated text (no JSON wrapper).
    """
    client = _make_bedrock_client()
    model_id = "mistral.mistral-7b-instruct-v0:2"

    # Mistral models on Bedrock expect the prompt wrapped in <s>[INST] ... [/INST]
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    payload: Dict[str, Any] = {
        "prompt": formatted_prompt,
        "max_tokens": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        response = client.invoke_model(
            body=json.dumps(payload),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        body_bytes = response["body"].read()
        result = json.loads(body_bytes)
        # Bedrock may return either a top‑level "generation" field or an "outputs" list.
        if isinstance(result, dict):
            # Direct generation field (used by some SDK examples)
            if "generation" in result:
                return result["generation"].strip()
            # Older style response with outputs list
            outputs = result.get("outputs")
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict):
                    # Prefer "text" if present
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"].strip()
                    # Fallback to "completion" or similar keys
                    for key in ("completion", "generated_text"):
                        if key in first and isinstance(first[key], str):
                            return first[key].strip()
        # If we get here, fall back to raw JSON string for debugging
        return json.dumps(result)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Bedrock call failed: {exc}") from exc