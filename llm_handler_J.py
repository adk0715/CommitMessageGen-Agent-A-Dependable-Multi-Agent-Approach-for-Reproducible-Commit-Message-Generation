# llm_handler.py (수정 완료된 버전)

import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import config

# bitsandbytes 라이브러리가 있는지 확인
try:
    import bitsandbytes as bnb # noqa: F401
    _BITSANDBYTES_AVAILABLE = True
    print("bitsandbytes available: enabling 4-bit when CUDA is present.")
except ImportError:
    _BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not available: 4-bit will be disabled.")

class LocalHFHandler:
    def __init__(self, model_key: str):
        info = config.MODELS[model_key]
        model_path = info["model_name"]
        gen_cfg = info.get("generation_config", {}) or {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[LocalHFHandler] Loading model: {model_path}")

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            # CHANGED: False로 변경하여 필요시 다운로드 허용
            local_files_only=False,
            padding_side="left",
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Model load (4-bit if possible on CUDA) ---
        quantization_config = None
        model_load_kwargs = {
            # CHANGED: False로 변경하여 필요시 다운로드 허용
            "local_files_only": False,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }
        if self.device == "cuda":
            model_load_kwargs["device_map"] = "auto"
            if _BITSANDBYTES_AVAILABLE:
                print(" -> Using 4-bit NF4 with bfloat16 compute.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs,
        )
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # --- Generation defaults ---
        do_sample = bool(gen_cfg.get("do_sample", False))
        self.generation_defaults = {
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 64)),
            "do_sample": do_sample,
            "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.05)),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            if "temperature" in gen_cfg:
                self.generation_defaults["temperature"] = float(gen_cfg["temperature"])
            if "top_p" in gen_cfg:
                self.generation_defaults["top_p"] = float(gen_cfg["top_p"])
            if "top_k" in gen_cfg:
                self.generation_defaults["top_k"] = int(gen_cfg["top_k"])

        print("[LocalHFHandler] Model & tokenizer loaded.")

    def _merge_and_sanitize_gen_kwargs(self, overrides: dict | None) -> GenerationConfig:
        merged = dict(self.generation_defaults)
        if overrides:
            if "max_tokens" in overrides and "max_new_tokens" not in overrides:
                overrides = dict(overrides)
                overrides["max_new_tokens"] = overrides.pop("max_tokens")
            merged.update(overrides)

        if merged.get("do_sample") is False:
            for k in ("temperature", "top_p", "top_k"):
                merged.pop(k, None)
        else:
            if "temperature" in merged and merged.get("temperature") is not None:
                try:
                    if float(merged["temperature"]) <= 1e-6:
                        merged.pop("temperature")
                except (ValueError, TypeError):
                    merged.pop("temperature")
        
        if "max_new_tokens" not in merged:
            merged["max_new_tokens"] = 64
            
        return GenerationConfig(**merged)

    def generate_batch(self, prompts, **overrides):
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if self.device == "cuda":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            gen_config = self._merge_and_sanitize_gen_kwargs(overrides)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            pad_id = self.tokenizer.pad_token_id
            input_ids = inputs["input_ids"]
            results = []
            for i in range(len(prompts)):
                prompt_len = int((input_ids[i] != pad_id).sum().item())
                gen_tokens = outputs[i][prompt_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                results.append(text.strip())
            return results

        except Exception as e:
            print(f"[LocalHFHandler] Batch generation failed: {e}")
            traceback.print_exc()
            return [""] * len(prompts)


def get_llm_handler(model_key: str):
    return LocalHFHandler(model_key)