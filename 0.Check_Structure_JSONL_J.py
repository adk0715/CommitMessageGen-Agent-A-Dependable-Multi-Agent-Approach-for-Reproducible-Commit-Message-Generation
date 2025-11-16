# =====================[ PATCH ❶: 최상단 환경 변수 (NumPy/rank_bm25 import 전에) ]=====================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# joblib 임시폴더 고정 (경고/청소 이슈 완화)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp/joblib")
os.makedirs(os.environ["JOBLIB_TEMP_FOLDER"], exist_ok=True)
# ================================================================================================

os.system("clear")

# =====================[ Import ]=================================================================
import glob
import json
import gc
import csv
import heapq
import pickle  # [추가] 체크포인트 저장을 위해 임포트
from typing import List, Tuple, Dict, Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed, parallel_backend

# =====================[ 설정 ]====================================================================
TOP_K = 10
OUTPUT_CSV_PATH = f"bm25_top_{TOP_K}_mapping_results_diff_F.csv"
CHECKPOINT_PATH = "bm25_checkpoint.pkl"  # [추가] 체크포인트 파일 경로

# 모드/태스크 토글
TYPE = ["diff", "commit"]
TYPE_TASK = TYPE[0]  # "diff" 또는 "commit"

# 개발 샘플링(0이면 전체)
DEV_LIMIT_DOCS = 0
DEV_LIMIT_QUERIES = 0

# === 안정 모드(너 환경 기준: RAM 125GiB / 32코어) ===
SHARD_DOCS = 300_000  # 샤드당 문서 수(50만 → 30만으로 축소)
N_WORKERS = 2  # 동시 워커(8 → 2로 축소)
MAX_TEXT_LEN = 2000  # CSV 저장 시 긴 텍스트 절단(0이면 무제한)
TRUNC_SUFFIX = " …<trunc>"

# 데이터 경로
corpus_files = [
    "../../1.Dataset/1.MCMD/clean/cpp/cpp_train_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/cpp/cpp_valid_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/csharp/csharp_train_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/csharp/csharp_valid_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/java/java_train_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/java/java_valid_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/python/python_train_combined_predictions2.jsonl",
    "../../1.Dataset/1.MCMD/clean/python/python_valid_combined_predictions2.jsonl",
    "../../1.Dataset/2.Commit-Chronicle/data/clean/train_combined_filtered.jsonl",
    "../../1.Dataset/2.Commit-Chronicle/data/clean/validation_combined_filtered.jsonl",
]
query_dir_path = "../../1.Dataset/3.MCMD+/"

# =====================[ 유틸 함수 ]===============================================================
def _get_sha(o: Dict[str, Any]) -> str:
    return (str(o.get("sha") or o.get("hash") or "")).strip()

def _maybe_trunc(s: str) -> str:
    if not s or MAX_TEXT_LEN <= 0:
        return s or ""
    return s if len(s) <= MAX_TEXT_LEN else s[:MAX_TEXT_LEN] + TRUNC_SUFFIX

def _pick_text_for_index(data: Dict[str, Any], is_commit_chronicle: bool) -> str:
    if TYPE_TASK == "diff":
        for key in ("diff", "raw_diff", "ref_diff", "processed_diff"):
            if key in data and data[key]:
                return data[key]
        msg_key = "message" if is_commit_chronicle else "clean"
        return data.get(msg_key, "")
    else:
        msg_key = "message" if is_commit_chronicle else "clean"
        return data.get(msg_key, "")

def _pick_text_for_query(data: Dict[str, Any]) -> str:
    if TYPE_TASK == "diff":
        for key in ("diff", "processed_diff", "raw_diff"):
            if key in data and data[key]:
                return data[key]
        for key in ("msg", "message", "clean"):
            if key in data and data[key]:
                return data[key]
        return ""
    else:
        for key in ("msg", "message", "clean"):
            if key in data and data[key]:
                return data[key]
        for key in ("diff", "processed_diff", "raw_diff"):
            if key in data and data[key]:
                return data[key]
        return ""

def iter_corpus_blocks(files: List[str], block_docs: int):
    block_tokens, block_meta = [], []
    produced = 0
    for file_path in files:
        is_commit_chronicle = "Commit-Chronicle" in file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _pick_text_for_index(data, is_commit_chronicle)
                    tokens = (text or "").split()
                    block_tokens.append(tokens)
                    meta = {
                        "source_file": os.path.basename(file_path),
                        "text": text,  # 최종 저장 시 _maybe_trunc로 절단
                        "sha": _get_sha(data),
                        "repo": data.get("repo"),
                    }
                    block_meta.append(meta)
                    produced += 1

                    if DEV_LIMIT_DOCS and produced >= DEV_LIMIT_DOCS:
                        if block_tokens:
                            yield block_tokens, block_meta
                        return

                    if block_docs and len(block_tokens) >= block_docs:
                        yield block_tokens, block_meta
                        block_tokens, block_meta = [], []
        except FileNotFoundError:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
    if block_tokens:
        yield block_tokens, block_meta

def load_and_tokenize_corpus_all(files: List[str]):
    tokenized_corpus, corpus_metadata = [], []
    for tokens, meta in iter_corpus_blocks(files, block_docs=0):
        tokenized_corpus.extend(tokens)
        corpus_metadata.extend(meta)
    return tokenized_corpus, corpus_metadata

def load_queries(dir_path: str):
    print("쿼리 데이터 로딩 시작...")
    query_files = glob.glob(os.path.join(dir_path, "*.jsonl"))
    queries = []
    taken = 0
    for file_path in tqdm(query_files, desc="쿼리 파일 로딩"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                queries.append(data)
                taken += 1
                if DEV_LIMIT_QUERIES and taken >= DEV_LIMIT_QUERIES:
                    break
        if DEV_LIMIT_QUERIES and taken >= DEV_LIMIT_QUERIES:
            break
    return queries

# =====================[ 저장 유틸: CSV 스트리밍 ]=================================================
CSV_FIELDS = [
    "query_sha", "query_msg", "rank", "similarity_score",
    "matched_source_file", "matched_sha", "matched_repo", "matched_msg"
]

def _open_csv(path, write_header: bool):
    # [수정] 'a' (append) 모드로 변경
    f = open(path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()
    return f, writer

# =====================[ 메인 로직 ]==============================================================
def main():
    queries = load_queries(query_dir_path)
    print(f"총 {len(queries)}개의 쿼리를 처리합니다.")

    query_records = []
    for q in queries:
        qtext = _pick_text_for_query(q)
        query_records.append(
            {
                "sha": _get_sha(q),
                "text": qtext,
                "tokens": (qtext or "").split(),
            }
        )

    # [수정] ===== 재개 로직 시작 =====
    resuming_from_checkpoint = os.path.exists(CHECKPOINT_PATH)
    resuming_from_csv = os.path.exists(OUTPUT_CSV_PATH)
    is_resuming = resuming_from_checkpoint or resuming_from_csv

    completed_query_shas = set()
    if resuming_from_csv:
        print(f"기존 출력 파일 {OUTPUT_CSV_PATH} 발견. 완료된 쿼리 로드 중...")
        try:
            with open(OUTPUT_CSV_PATH, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "query_sha" in row:
                        completed_query_shas.add(row["query_sha"])
            print(f"총 {len(completed_query_shas)}개의 쿼리 SHA를 로드했습니다. (최종 쓰기 시 건너뜀)")
        except Exception as e:
            print(f"경고: 기존 CSV 파일 읽기 실패: {e}. 덧쓰기 모드로 전환합니다.")
            is_resuming = False
            completed_query_shas.clear()
            if os.path.exists(OUTPUT_CSV_PATH):
                os.remove(OUTPUT_CSV_PATH)

    if not is_resuming:
        print("새로운 실행을 시작합니다. 기존 출력 파일/체크포인트(있다면)를 삭제합니다.")
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    # [수정] CSV는 항상 'a' 모드로 열고, 'is_resuming'이 아닐 때만 헤더를 씁니다.
    csv_f, csv_w = _open_csv(OUTPUT_CSV_PATH, write_header=not is_resuming)
    # ===== 재개 로직 끝 =====

    if SHARD_DOCS <= 0:
        print("모든 코퍼스를 메모리에 로드합니다 (메모리 사용량 높음)")
        tokenized_corpus, corpus_metadata = load_and_tokenize_corpus_all(corpus_files)
        print(f"\n총 {len(tokenized_corpus):,}개의 문서로 BM25 인덱스를 생성합니다...")
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 인덱스 생성 완료.")
        del tokenized_corpus
        gc.collect()

        # [수정] 이미 완료된 쿼리는 건너뛰도록 수정
        print("쿼리 처리를 시작합니다. (완료된 쿼리는 건너뜁니다)")
        processed_count = 0
        for qi, qrec in enumerate(tqdm(query_records, desc="쿼리 처리 중")):
            if qrec["sha"] in completed_query_shas:  # [수정]
                continue

            if not qrec["tokens"]:
                continue
            doc_scores = bm25.get_scores(qrec["tokens"])
            top_k_indices = np.argsort(doc_scores)[::-1][:TOP_K]
            for rank, idx in enumerate(top_k_indices, 1):
                matched_doc = corpus_metadata[idx]
                csv_w.writerow(
                    {
                        "query_sha": qrec["sha"],
                        "query_msg": _maybe_trunc(qrec["text"]),
                        "rank": rank,
                        "similarity_score": float(doc_scores[idx]),
                        "matched_source_file": matched_doc["source_file"],
                        "matched_sha": matched_doc.get("sha"),
                        "matched_repo": matched_doc.get("repo"),
                        "matched_msg": _maybe_trunc(matched_doc["text"]),
                    }
                )
            
            processed_count += 1
            if processed_count % 50 == 0: # [수정]
                csv_f.flush()
        
        # cleanup
        del bm25, corpus_metadata
        gc.collect()

    else:
        print(f"코퍼스를 {SHARD_DOCS}개 단위의 샤드로 나누어 처리합니다 (메모리 효율적)")
        
        # [수정] ===== 체크포인트 로드 시작 =====
        start_block_id = 0  # 0이면 1번 블록부터 시작 (1-based)
        per_query_heaps: Dict[int, List[Tuple[float, Tuple[int, int], Dict[str, Any]]]] = \
            {qi: [] for qi in range(len(query_records))}

        if resuming_from_checkpoint:
            print(f"Checkpoint file '{CHECKPOINT_PATH}' 발견. 로드 시도 중...")
            try:
                with open(CHECKPOINT_PATH, "rb") as f:
                    start_block_id, per_query_heaps = pickle.load(f)
                
                # 쿼리 수 검증
                if len(per_query_heaps) != len(query_records):
                    raise ValueError(f"Checkpoint의 쿼리 수({len(per_query_heaps)})가 현재 쿼리 수({len(query_records)})와 불일치.")
                
                print(f"Checkpoint 로드 완료. 블록 {start_block_id + 1}부터 재시작합니다.")
            except Exception as e:
                print(f"Checkpoint 로드 실패: {e}. 블록 1부터 다시 처리합니다.")
                if os.path.exists(CHECKPOINT_PATH):
                    os.remove(CHECKPOINT_PATH) # 손상된 파일 삭제
                start_block_id = 0
                per_query_heaps = {qi: [] for qi in range(len(query_records))}
        else:
            print("Checkpoint 없음. 블록 1부터 시작합니다.")
        # ===== 체크포인트 로드 끝 =====


        # [수정] ===== 블록 건너뛰기 로직 시작 =====
        block_id_counter = 0  # 실제 처리된 블록 카운터 (0-based)
        block_iterator = iter_corpus_blocks(corpus_files, SHARD_DOCS)

        if start_block_id > 0:
            print(f"{start_block_id}개의 완료된 블록을 건너뜁니다...")
            for _ in tqdm(range(start_block_id), desc="Skipping blocks"):
                try:
                    next(block_iterator)
                    block_id_counter += 1
                except StopIteration:
                    print("경고: Checkpoint가 이전 실행의 끝을 가리킵니다. 건너뛰기 중지.")
                    break
            print(f"총 {block_id_counter}개 블록 건너뜀. {block_id_counter + 1}부터 시작.")
        # ===== 블록 건너뛰기 로직 끝 =====

        # [수정] block_id -> block_id_counter로 변수명 변경 및 로깅 수정
        for block_tokens, block_meta in block_iterator:
            block_id_counter += 1 # 현재 블록 번호 (1-based)
            
            print(f"[샤딩] 블록 {block_id_counter} 시작 (문서 {len(block_tokens)}개)")
            bm25_block = BM25Okapi(block_tokens)

            def _topk_for_query(qi: int, qrec: Dict[str, Any]):
                if not qrec["tokens"]:
                    return qi, np.array([], dtype=int), np.array([], dtype=float)
                scores = bm25_block.get_scores(qrec["tokens"])
                if len(scores) <= TOP_K:
                    idx = np.argsort(scores)[::-1]
                else:
                    idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
                    idx = idx[np.argsort(scores[idx])[::-1]]
                return qi, idx.astype(int), scores[idx].astype(float)

            # 내부 스레드 1개로 제한하여 과다 병렬화 방지
            with parallel_backend("loky", inner_max_num_threads=1):
                pairs = Parallel(
                    n_jobs=N_WORKERS,
                    backend="loky",
                    batch_size=32,
                    verbose=10,
                )(
                    delayed(_topk_for_query)(qi, qrec)
                    for qi, qrec in enumerate(query_records)
                )

            # 메인 프로세스에서만 메타 조회 후 힙에 병합
            for qi, idx_arr, score_arr in tqdm(pairs, desc=f"블록 {block_id_counter} 결과 병합"):
                heap = per_query_heaps[qi]
                for bi, sc in zip(idx_arr.tolist(), score_arr.tolist()):
                    # [수정] block_id 대신 block_id_counter 사용
                    item = (float(sc), (block_id_counter, int(bi)), block_meta[int(bi)])
                    if len(heap) < TOP_K:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)

            # 블록 객체 정리
            del pairs, bm25_block, block_tokens, block_meta
            gc.collect()
            print(f"[샤딩] 블록 {block_id_counter} 완료")

            # [추가] ===== 매 블록 완료 시 체크포인트 저장 =====
            print(f"Checkpoint 저장 중 (블록 {block_id_counter} 완료)...")
            try:
                with open(CHECKPOINT_PATH, "wb") as f:
                    pickle.dump((block_id_counter, per_query_heaps), f)
                print(f"Checkpoint {CHECKPOINT_PATH}에 저장 완료.")
            except Exception as e:
                print(f"경고: Checkpoint 저장 실패: {e}")
            # ===== 체크포인트 저장 끝 =====

        # 통합 Top-K → 즉시 CSV 기록
        print("모든 블록 처리 완료. 최종 Top-K 결과를 CSV에 저장합니다...") # [수정]
        for qi, heap in enumerate(tqdm(per_query_heaps.values(), desc="최종 결과 저장")): # [수정]
            qsha = query_records[qi]["sha"]
            
            # [수정] 이미 CSV에 기록된 쿼리는 건너뜀
            if qsha in completed_query_shas:
                continue
                
            qtext = query_records[qi]["text"]
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
            for rank, (score, _block_idx, meta) in enumerate(sorted_items, start=1):
                csv_w.writerow(
                    {
                        "query_sha": qsha,
                        "query_msg": _maybe_trunc(qtext),
                        "rank": rank,
                        "similarity_score": float(score),
                        "matched_source_file": meta["source_file"],
                        "matched_sha": meta.get("sha"),
                        "matched_repo": meta.get("repo"),
                        "matched_msg": _maybe_trunc(meta["text"]),
                    }
                )
            if qi % 50 == 0:
                csv_f.flush()

        # [추가] ===== 최종 완료 후 체크포인트 삭제 =====
        print("최종 CSV 쓰기 완료. Checkpoint 파일을 삭제합니다.")
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        # ===== 체크포인트 삭제 끝 =====

        # cleanup
        del per_query_heaps
        gc.collect()

    # =====================[ 끝 ]================================================================
    csv_f.flush()
    csv_f.close()
    print("모든 작업이 완료되었습니다.")
    print(f"결과가 {OUTPUT_CSV_PATH}에 저장되었습니다.")

# =====================[ Entry ]==================================================================
if __name__ == "__main__":
    main()