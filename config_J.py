# config.py (수정 완료)
"""
프로젝트의 모든 설정을 중앙에서 관리합니다.
API 키, 파일 경로, 모델 파라미터 등을 여기서 수정하세요.
"""
import os
import glob

# --- API 키 설정 ---
API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
}

# --- 기본 경로 설정 ---
DATA_DIR = "./"

# --- HF 캐시 루트 & 스냅샷 해석 ---
HF_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hf_cache"))

# NOTE: 아래 함수는 로컬 캐시가 '반드시' 존재할 때만 유효합니다.
# 자동 다운로드를 위해서는 이 함수를 사용하는 대신 Hub ID를 직접 사용해야 합니다.
def _resolve_hf_snapshot(local_repo_dir: str) -> str:
    snap_glob = os.path.join(local_repo_dir, "snapshots", "*")
    candidates = glob.glob(snap_glob)
    if not candidates:
        return local_repo_dir
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# --- 언어별 파일 경로 설정 ---
PATHS = {
    "py": {
        "main_data": os.path.join(DATA_DIR, "py.jsonl"),
        "few_shot_examples": os.path.join(DATA_DIR, "pybest_no_selectv.jsonl"),
        "output_dir": os.path.join(DATA_DIR, "results/python"),
    },
    "java": {
        "main_data": os.path.join(DATA_DIR, "java.jsonl"),
        "few_shot_examples": os.path.join(DATA_DIR, "javabest_no_select.jsonl"),
        "output_dir": os.path.join(DATA_DIR, "results/java"),
    },
}

# --- 로컬 모델 경로(참고용으로 유지) ---
_HUGGY_LLAMA7B_DIR  = os.path.join(HF_CACHE_DIR, "models--huggyllama--llama-7b")
_HUGGY_LLAMA7B_PATH = _resolve_hf_snapshot(_HUGGY_LLAMA7B_DIR)


# --- 모델별 설정 ---
# --- 모델별 설정 ---
MODELS = {
    "gemini": {
        "model_name": "gemini-pro",
        "generation_config": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_output_tokens": 50,
        },
    },
    "gpt": {
        "model_name": "gpt-3.5-turbo-16k",
        "generation_config": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 50,
            "n": 1,
        },
    },

    # ---- LLaMA v2 7B (huggyllama 미러)
    "llama_7b": {
        "model_name": "huggyllama/llama-7b",
        "hf_cache_dir": HF_CACHE_DIR,
        "generation_config": {
            "max_new_tokens": 28,
            "do_sample": False,
            "repetition_penalty": 1.07,
        },
    },

    "llama2_13b": {
        "model_name": "../../LLM_Models/Llama/v2/13B",
        "generation_config": {
            "max_new_tokens": 50,
            "do_sample": False,
            "repetition_penalty": 1.07,
        },
    },

    # ---- Qwen 1.5 7B Chat
    "qwen_7b_chat": {
        "model_name": "Qwen/Qwen1.5-7B-Chat",
        "hf_cache_dir": HF_CACHE_DIR,
        "trust_remote_code": True,
        "generation_config": {
            "max_new_tokens": 48,
            "do_sample": False,
            "top_p": 1.0,
            "temperature": 0.0,
            "repetition_penalty": 1.07,
        },
        "runtime_hints": {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "auto",
        },
    },
}


# --- 실행 관련 설정 ---
MAX_RETRIES = 5
RETRY_DELAY = 2