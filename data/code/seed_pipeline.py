from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("seed_pipeline_config.yaml")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "seed_pipeline"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

TOKEN_RE = re.compile(r"\b[\w-]{2,}\b", re.UNICODE)
WS_RE = re.compile(r"\s+")


@dataclass
class DomainPaths:
    base: Path
    seed_urls: Path
    discovered_urls: Path
    candidate_chunks: Path
    seed_chunks: Path
    labels: Path
    model: Path
    model_info: Path
    fasttext_train: Path
    fasttext_model: Path
    scored_chunks: Path
    run_log: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def canonicalize_url(url: str) -> str:
    clean, _ = urldefrag(url.strip())
    parsed = urlparse(clean)
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def load_jsonl_ids(path: Path, field: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            val = obj.get(field)
            if isinstance(val, str) and val:
                ids.add(val)
    return ids


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def safe_logit_from_prob(prob: float, eps: float = 1e-6) -> float:
    p = min(max(prob, eps), 1.0 - eps)
    return math.log(p) - math.log(1.0 - p)


def load_fasttext_module():
    try:
        import fasttext  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "fasttext is not installed. Install it with: python -m pip install fasttext-wheel"
        ) from exc
    return fasttext


def domain_paths(data_root: Path, domain: str) -> DomainPaths:
    base = data_root / "domains" / domain
    return DomainPaths(
        base=base,
        seed_urls=base / "seed_urls.jsonl",
        discovered_urls=base / "discovered_urls.jsonl",
        candidate_chunks=base / "candidate_chunks.jsonl",
        seed_chunks=base / "seed_chunks.jsonl",
        labels=base / "labels.jsonl",
        model=base / "discriminator_model.json",
        model_info=base / "discriminator_model_info.json",
        fasttext_train=base / "fasttext_train.txt",
        fasttext_model=base / "fasttext_model.bin",
        scored_chunks=base / "scored_chunks.jsonl",
        run_log=base / "run_log.jsonl",
    )


def resolve_config_and_domain(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = load_yaml(config_path)
    domains = config.get("domains", {})
    if not isinstance(domains, dict):
        raise ValueError("Config must contain a mapping at `domains`.")
    if args.domain not in domains:
        raise ValueError(f"Unknown domain `{args.domain}`. Available: {sorted(domains)}")
    domain_cfg = domains[args.domain]
    if not isinstance(domain_cfg, dict):
        raise ValueError(f"Domain config for `{args.domain}` must be a mapping.")

    configured_data_root = config.get("data_root")
    data_root = Path(configured_data_root) if configured_data_root else DEFAULT_DATA_ROOT
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    return config, domain_cfg, data_root


def get_default_cfg(config: dict, section: str, key: str, default):
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        return default
    sec = defaults.get(section, {})
    if not isinstance(sec, dict):
        return default
    return sec.get(key, default)


def keyword_rule_signal(text: str, include_keywords: list[str], exclude_keywords: list[str]) -> tuple[int, int]:
    lowered = text.lower()
    inc = sum(1 for kw in include_keywords if kw.lower() in lowered)
    exc = sum(1 for kw in exclude_keywords if kw.lower() in lowered)
    return inc, exc


def split_words(words: list[str], chunk_words: int, overlap_words: int):
    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    step = max(1, chunk_words - max(0, overlap_words))
    i = 0
    n = len(words)
    while i < n:
        yield words[i : i + chunk_words]
        i += step


def extract_text_from_html(html: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = normalize_ws(soup.title.string)

    text = normalize_ws(soup.get_text(" "))
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if isinstance(href, str):
            links.append(href.strip())
    return title, text, links


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 40) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    for idx in range(min(len(reader.pages), max_pages)):
        page_txt = normalize_ws(reader.pages[idx].extract_text() or "")
        if page_txt:
            parts.append(page_txt)
    return normalize_ws(" ".join(parts))


def fetch_document(session: requests.Session, url: str, timeout_sec: int, user_agent: str) -> dict | None:
    headers = {"User-Agent": user_agent}
    try:
        resp = session.get(url, timeout=timeout_sec, headers=headers, allow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    final_url = canonicalize_url(resp.url or url)
    content_type = (resp.headers.get("Content-Type") or "").lower()

    if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
        try:
            text = extract_text_from_pdf_bytes(resp.content)
        except Exception:
            return None
        if not text:
            return None
        return {
            "url": final_url,
            "title": Path(urlparse(final_url).path).name,
            "text": text,
            "links": [],
            "content_type": "application/pdf",
            "status_code": resp.status_code,
        }

    try:
        html = resp.text
    except Exception:
        return None
    title, text, links = extract_text_from_html(html)
    if not text:
        return None
    return {
        "url": final_url,
        "title": title,
        "text": text,
        "links": links,
        "content_type": content_type or "text/html",
        "status_code": resp.status_code,
    }


def to_absolute_links(base_url: str, links: list[str]) -> list[str]:
    out: list[str] = []
    for link in links:
        if not link:
            continue
        absolute = urljoin(base_url, link)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        out.append(canonicalize_url(absolute))
    return out


def host_from_url(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def url_allowed(url: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return True
    host = host_from_url(url)
    for dom in allowed_domains:
        d = dom.lower().strip()
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def init_domain(args: argparse.Namespace) -> int:
    _, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    seeded = load_jsonl_ids(paths.seed_urls, "url")
    added = 0
    for raw_url in domain_cfg.get("seed_urls", []) or []:
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = canonicalize_url(raw_url)
        if url in seeded:
            continue
        append_jsonl(
            paths.seed_urls,
            {
                "url": url,
                "domain": args.domain,
                "source": "config_seed_url",
                "created_at": now_iso(),
            },
        )
        seeded.add(url)
        added += 1

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "init",
            "seed_urls_added": added,
            "seed_urls_total": len(seeded),
        },
    )
    print("Initialized domain:", args.domain)
    print("Seed URLs added:", added)
    print("Seed URL file:", paths.seed_urls)
    return 0


def discover_urls(args: argparse.Namespace) -> int:
    config, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    max_depth = args.max_depth if args.max_depth is not None else int(get_default_cfg(config, "crawl", "max_depth", 1))
    max_pages = args.max_pages if args.max_pages is not None else int(
        get_default_cfg(config, "crawl", "max_pages_per_iteration", 80)
    )
    timeout_sec = args.timeout if args.timeout is not None else int(
        get_default_cfg(config, "crawl", "request_timeout_sec", 20)
    )
    delay_sec = args.delay if args.delay is not None else float(get_default_cfg(config, "crawl", "delay_sec", 0.5))
    allowed_domains = list(domain_cfg.get("allowed_domains", []) or [])

    known_urls = load_jsonl_ids(paths.discovered_urls, "url")
    seed_rows = load_jsonl(paths.seed_urls)
    if not seed_rows:
        raise ValueError(f"No seed URLs found for `{args.domain}`. Run `init` first.")

    q: deque[tuple[str, int, str]] = deque()
    seen_in_run: set[str] = set()
    for row in seed_rows:
        raw = row.get("url")
        if isinstance(raw, str) and raw:
            q.append((canonicalize_url(raw), 0, "seed"))

    fetched = 0
    discovered = 0
    session = requests.Session()
    user_agent = args.user_agent or DEFAULT_USER_AGENT

    while q and fetched < max_pages:
        current_url, depth, parent_url = q.popleft()
        if current_url in seen_in_run or not url_allowed(current_url, allowed_domains):
            continue
        seen_in_run.add(current_url)

        doc = fetch_document(session, current_url, timeout_sec=timeout_sec, user_agent=user_agent)
        fetched += 1
        if not doc:
            time.sleep(delay_sec)
            continue

        if current_url not in known_urls:
            append_jsonl(
                paths.discovered_urls,
                {
                    "url": current_url,
                    "domain": args.domain,
                    "parent_url": parent_url,
                    "depth": depth,
                    "status_code": doc.get("status_code"),
                    "content_type": doc.get("content_type"),
                    "discovered_at": now_iso(),
                },
            )
            known_urls.add(current_url)
            discovered += 1

        if depth < max_depth:
            for nxt in to_absolute_links(doc["url"], doc.get("links", [])):
                if nxt not in seen_in_run and url_allowed(nxt, allowed_domains):
                    q.append((nxt, depth + 1, current_url))
        time.sleep(delay_sec)

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "discover",
            "fetched_pages": fetched,
            "new_urls": discovered,
            "max_depth": max_depth,
            "max_pages": max_pages,
        },
    )
    print("Domain:", args.domain)
    print("Fetched pages:", fetched)
    print("New discovered URLs:", discovered)
    print("Discovered URL file:", paths.discovered_urls)
    return 0


def ingest_candidates(args: argparse.Namespace) -> int:
    config, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    timeout_sec = args.timeout if args.timeout is not None else int(
        get_default_cfg(config, "crawl", "request_timeout_sec", 20)
    )
    delay_sec = args.delay if args.delay is not None else float(get_default_cfg(config, "crawl", "delay_sec", 0.5))
    chunk_words = args.chunk_words if args.chunk_words is not None else int(
        get_default_cfg(config, "ingest", "chunk_words", 220)
    )
    overlap_words = args.chunk_overlap if args.chunk_overlap is not None else int(
        get_default_cfg(config, "ingest", "chunk_overlap_words", 40)
    )
    min_chars = args.min_chars if args.min_chars is not None else int(
        get_default_cfg(config, "ingest", "min_text_chars", 500)
    )
    max_chunks = args.max_chunks if args.max_chunks is not None else int(
        get_default_cfg(config, "ingest", "max_chunks_per_doc", 25)
    )

    include_keywords = list(domain_cfg.get("include_keywords", []) or [])
    exclude_keywords = list(domain_cfg.get("exclude_keywords", []) or [])
    allowed_domains = list(domain_cfg.get("allowed_domains", []) or [])

    known_chunk_ids = load_jsonl_ids(paths.candidate_chunks, "chunk_id")
    discovered = load_jsonl(paths.discovered_urls)
    if not discovered:
        raise ValueError(f"No discovered URLs found for `{args.domain}`. Run `discover` first.")

    todo_urls = []
    for row in discovered:
        raw = row.get("url")
        if isinstance(raw, str) and raw:
            u = canonicalize_url(raw)
            if url_allowed(u, allowed_domains):
                todo_urls.append(u)
    if args.max_urls is not None:
        todo_urls = todo_urls[: args.max_urls]

    session = requests.Session()
    user_agent = args.user_agent or DEFAULT_USER_AGENT
    docs_ingested = 0
    chunks_written = 0

    for url in todo_urls:
        doc = fetch_document(session, url, timeout_sec=timeout_sec, user_agent=user_agent)
        if not doc:
            time.sleep(delay_sec)
            continue
        text = normalize_ws(doc["text"])
        if len(text) < min_chars:
            time.sleep(delay_sec)
            continue

        words = text.split()
        emitted = 0
        for chunk_idx, cwords in enumerate(split_words(words, chunk_words, overlap_words)):
            if emitted >= max_chunks:
                break
            chunk_text = " ".join(cwords).strip()
            if len(chunk_text) < max(120, min_chars // 3):
                continue

            chunk_id = sha1_text(f"{doc['url']}::{chunk_idx}::{chunk_text[:180]}")
            if chunk_id in known_chunk_ids:
                continue

            include_hits, exclude_hits = keyword_rule_signal(chunk_text, include_keywords, exclude_keywords)
            append_jsonl(
                paths.candidate_chunks,
                {
                    "chunk_id": chunk_id,
                    "domain": args.domain,
                    "url": doc["url"],
                    "host": host_from_url(doc["url"]),
                    "title": doc.get("title", ""),
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "content_type": doc.get("content_type", ""),
                    "include_hits": include_hits,
                    "exclude_hits": exclude_hits,
                    "ingested_at": now_iso(),
                },
            )
            known_chunk_ids.add(chunk_id)
            emitted += 1
            chunks_written += 1

        docs_ingested += 1
        time.sleep(delay_sec)

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "ingest",
            "docs_attempted": len(todo_urls),
            "docs_ingested": docs_ingested,
            "new_chunks": chunks_written,
        },
    )
    print("Domain:", args.domain)
    print("Docs attempted:", len(todo_urls))
    print("Docs ingested:", docs_ingested)
    print("New candidate chunks:", chunks_written)
    print("Candidate chunk file:", paths.candidate_chunks)
    return 0


def bootstrap_labels(args: argparse.Namespace) -> int:
    _, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    include_keywords = list(domain_cfg.get("include_keywords", []) or [])
    exclude_keywords = list(domain_cfg.get("exclude_keywords", []) or [])
    if not include_keywords or not exclude_keywords:
        raise ValueError("Domain must define both include_keywords and exclude_keywords.")

    existing = load_jsonl_ids(paths.labels, "chunk_id")
    candidates = load_jsonl(paths.candidate_chunks)
    if not candidates:
        raise ValueError(f"No candidate chunks for `{args.domain}`. Run `ingest` first.")

    pos = 0
    neg = 0
    for row in candidates:
        cid = row.get("chunk_id")
        text = row.get("text", "")
        if not isinstance(cid, str) or not cid or cid in existing:
            continue
        if not isinstance(text, str) or len(text) < 40:
            continue

        include_hits, exclude_hits = keyword_rule_signal(text, include_keywords, exclude_keywords)
        label = None
        reason = ""
        if include_hits >= args.min_include_hits and exclude_hits == 0:
            label, reason = 1, "weak_positive_keyword_rule"
        elif exclude_hits >= args.min_exclude_hits and include_hits == 0:
            label, reason = 0, "weak_negative_keyword_rule"

        if label is None:
            continue
        if label == 1 and pos >= args.max_pos:
            continue
        if label == 0 and neg >= args.max_neg:
            continue

        append_jsonl(
            paths.labels,
            {
                "chunk_id": cid,
                "domain": args.domain,
                "url": row.get("url", ""),
                "text": text,
                "label": label,
                "label_source": reason,
                "created_at": now_iso(),
            },
        )
        existing.add(cid)
        if label == 1:
            pos += 1
        else:
            neg += 1
        if pos >= args.max_pos and neg >= args.max_neg:
            break

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "bootstrap_labels",
            "positives_added": pos,
            "negatives_added": neg,
            "labels_file": str(paths.labels),
        },
    )
    print("Domain:", args.domain)
    print("Positive labels added:", pos)
    print("Negative labels added:", neg)
    print("Labels file:", paths.labels)
    return 0


def train_discriminator(args: argparse.Namespace) -> int:
    _, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(paths.labels)
    if not rows:
        raise ValueError(f"No labels found for `{args.domain}`: {paths.labels}")

    backend = str(getattr(args, "backend", "naive_bayes")).strip().lower()
    include_keywords = list(domain_cfg.get("include_keywords", []) or [])
    exclude_keywords = list(domain_cfg.get("exclude_keywords", []) or [])

    if backend == "fasttext":
        fasttext = load_fasttext_module()

        train_lines = 0
        pos_docs = 0
        neg_docs = 0
        ensure_parent(paths.fasttext_train)
        with paths.fasttext_train.open("w", encoding="utf-8") as f:
            for row in rows:
                label = row.get("label")
                text = row.get("text", "")
                if label not in {0, 1} or not isinstance(text, str):
                    continue
                cleaned = normalize_ws(text)
                if len(cleaned) < 20:
                    continue
                f.write(f"__label__{label} {cleaned}\n")
                train_lines += 1
                if label == 1:
                    pos_docs += 1
                else:
                    neg_docs += 1

        if pos_docs == 0 or neg_docs == 0:
            raise ValueError("Need at least one positive and one negative label to train.")

        model_ft = fasttext.train_supervised(
            input=str(paths.fasttext_train),
            lr=float(args.ft_lr),
            epoch=int(args.ft_epoch),
            wordNgrams=int(args.ft_word_ngrams),
            dim=int(args.ft_dim),
            minn=int(args.ft_minn),
            maxn=int(args.ft_maxn),
            loss=str(args.ft_loss),
            thread=int(args.ft_thread),
        )
        ensure_parent(paths.fasttext_model)
        model_ft.save_model(str(paths.fasttext_model))

        model_info = {
            "domain": args.domain,
            "backend": "fasttext",
            "trained_at": now_iso(),
            "model_path": str(paths.fasttext_model),
            "include_keywords": include_keywords,
            "exclude_keywords": exclude_keywords,
            "training_stats": {
                "pos_docs": pos_docs,
                "neg_docs": neg_docs,
                "labels_total": len(rows),
                "train_lines": train_lines,
            },
            "fasttext_params": {
                "lr": float(args.ft_lr),
                "epoch": int(args.ft_epoch),
                "wordNgrams": int(args.ft_word_ngrams),
                "dim": int(args.ft_dim),
                "minn": int(args.ft_minn),
                "maxn": int(args.ft_maxn),
                "loss": str(args.ft_loss),
                "thread": int(args.ft_thread),
            },
        }
        paths.model_info.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")

        append_jsonl(
            paths.run_log,
            {
                "at": now_iso(),
                "domain": args.domain,
                "step": "train",
                "backend": "fasttext",
                "labels_total": len(rows),
                "pos_docs": pos_docs,
                "neg_docs": neg_docs,
                "model_path": str(paths.fasttext_model),
            },
        )
        print("Domain:", args.domain)
        print("Backend:", "fasttext")
        print("Labels used:", len(rows))
        print("Positive docs:", pos_docs)
        print("Negative docs:", neg_docs)
        print("Model file:", paths.fasttext_model)
        return 0

    min_token_len = max(2, args.min_token_len)
    alpha = max(1e-6, args.alpha)
    min_df = max(1, args.min_df)
    max_vocab = max(100, args.max_vocab)

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    pos_docs = 0
    neg_docs = 0

    for row in rows:
        label = row.get("label")
        text = row.get("text", "")
        if label not in {0, 1} or not isinstance(text, str):
            continue
        toks = [t for t in tokenize(text) if len(t) >= min_token_len]
        if not toks:
            continue
        for t in set(toks):
            doc_freq[t] += 1
        if label == 1:
            pos_docs += 1
            pos_counts.update(toks)
        else:
            neg_docs += 1
            neg_counts.update(toks)

    if pos_docs == 0 or neg_docs == 0:
        raise ValueError("Need at least one positive and one negative label to train.")

    combined = pos_counts + neg_counts
    vocab = [tok for tok, df in doc_freq.items() if df >= min_df and len(tok) >= min_token_len]
    vocab.sort(key=lambda t: combined[t], reverse=True)
    vocab = vocab[:max_vocab]
    v = len(vocab)
    if v == 0:
        raise ValueError("Vocabulary is empty after filtering. Lower min_df/min_token_len.")

    pos_total = sum(pos_counts[t] for t in vocab)
    neg_total = sum(neg_counts[t] for t in vocab)

    token_log_odds: dict[str, float] = {}
    for tok in vocab:
        p_pos = (pos_counts[tok] + alpha) / (pos_total + alpha * v)
        p_neg = (neg_counts[tok] + alpha) / (neg_total + alpha * v)
        token_log_odds[tok] = math.log(p_pos) - math.log(p_neg)

    class_prior = pos_docs / (pos_docs + neg_docs)
    prior_logit = math.log(class_prior) - math.log(1.0 - class_prior)

    model = {
        "domain": args.domain,
        "trained_at": now_iso(),
        "backend": "naive_bayes",
        "class_prior": class_prior,
        "prior_logit": prior_logit,
        "alpha": alpha,
        "min_df": min_df,
        "max_vocab": max_vocab,
        "min_token_len": min_token_len,
        "vocab_size": v,
        "token_log_odds": token_log_odds,
        "include_keywords": include_keywords,
        "exclude_keywords": exclude_keywords,
        "keyword_boost": args.keyword_boost,
        "training_stats": {
            "pos_docs": pos_docs,
            "neg_docs": neg_docs,
            "labels_total": len(rows),
        },
    }

    ensure_parent(paths.model)
    paths.model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.model_info.write_text(
        json.dumps(
            {
                "domain": args.domain,
                "backend": "naive_bayes",
                "trained_at": now_iso(),
                "model_path": str(paths.model),
                "include_keywords": include_keywords,
                "exclude_keywords": exclude_keywords,
                "keyword_boost": float(args.keyword_boost),
                "training_stats": {
                    "pos_docs": pos_docs,
                    "neg_docs": neg_docs,
                    "labels_total": len(rows),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "train",
            "backend": "naive_bayes",
            "labels_total": len(rows),
            "pos_docs": pos_docs,
            "neg_docs": neg_docs,
            "vocab_size": v,
            "model_path": str(paths.model),
        },
    )
    print("Domain:", args.domain)
    print("Backend:", "naive_bayes")
    print("Labels used:", len(rows))
    print("Positive docs:", pos_docs)
    print("Negative docs:", neg_docs)
    print("Vocab size:", v)
    print("Model file:", paths.model)
    return 0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_text_with_model(text: str, model: dict) -> tuple[float, float, dict]:
    toks = tokenize(text)
    token_log_odds: dict[str, float] = model.get("token_log_odds", {})
    logit = float(model.get("prior_logit", 0.0))

    token_matches = 0
    for t in toks:
        if t in token_log_odds:
            logit += float(token_log_odds[t])
            token_matches += 1

    include_keywords = list(model.get("include_keywords", []) or [])
    exclude_keywords = list(model.get("exclude_keywords", []) or [])
    include_hits, exclude_hits = keyword_rule_signal(text, include_keywords, exclude_keywords)
    rule_signal = include_hits - exclude_hits
    logit += float(model.get("keyword_boost", 0.0)) * rule_signal

    return sigmoid(logit), logit, {
        "token_matches": token_matches,
        "include_hits": include_hits,
        "exclude_hits": exclude_hits,
    }


def score_candidates(args: argparse.Namespace) -> int:
    _, domain_cfg, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    model_info: dict = {}
    if paths.model_info.exists():
        model_info = json.loads(paths.model_info.read_text(encoding="utf-8"))

    backend_arg = getattr(args, "backend", None)
    if backend_arg:
        backend = str(backend_arg).strip().lower()
    elif model_info.get("backend"):
        backend = str(model_info.get("backend")).strip().lower()
    elif paths.fasttext_model.exists():
        backend = "fasttext"
    elif paths.model.exists():
        backend = "naive_bayes"
    else:
        raise FileNotFoundError(
            f"No discriminator model found for `{args.domain}`. Run `train` first."
        )

    model = None
    fasttext_model = None
    if backend == "naive_bayes":
        if not paths.model.exists():
            raise FileNotFoundError(f"Naive Bayes model not found: {paths.model}.")
        model = json.loads(paths.model.read_text(encoding="utf-8"))
    elif backend == "fasttext":
        if not paths.fasttext_model.exists():
            raise FileNotFoundError(f"fastText model not found: {paths.fasttext_model}.")
        fasttext = load_fasttext_module()
        fasttext_model = fasttext.load_model(str(paths.fasttext_model))
    else:
        raise ValueError(f"Unsupported backend for scoring: {backend}")

    include_keywords = list(model_info.get("include_keywords") or domain_cfg.get("include_keywords", []) or [])
    exclude_keywords = list(model_info.get("exclude_keywords") or domain_cfg.get("exclude_keywords", []) or [])

    candidates = load_jsonl(paths.candidate_chunks)
    if not candidates:
        raise ValueError(f"No candidate chunks found: {paths.candidate_chunks}")

    existing = load_jsonl_ids(paths.scored_chunks, "chunk_id")
    accept_t = args.accept_threshold
    reject_t = args.reject_threshold
    if reject_t >= accept_t:
        raise ValueError("reject_threshold must be less than accept_threshold.")

    counts = {"accept": 0, "review": 0, "reject": 0}
    new_scores = 0

    for row in candidates:
        cid = row.get("chunk_id")
        text = row.get("text")
        if not isinstance(cid, str) or cid in existing:
            continue
        if not isinstance(text, str) or len(text) < 20:
            continue

        if backend == "naive_bayes":
            assert model is not None
            prob, logit, dbg = score_text_with_model(text, model)
        else:
            assert fasttext_model is not None
            cleaned = normalize_ws(text)
            labels, probs = fasttext_model.predict(cleaned, k=2)
            prob_map = {lbl: float(p) for lbl, p in zip(labels, probs)}
            if "__label__1" in prob_map:
                prob = prob_map["__label__1"]
            elif "__label__0" in prob_map:
                prob = 1.0 - prob_map["__label__0"]
            else:
                label1, prob1 = fasttext_model.predict(cleaned, k=1)
                top_label = label1[0] if label1 else "__label__0"
                top_prob = float(prob1[0]) if prob1 else 0.5
                prob = top_prob if top_label == "__label__1" else (1.0 - top_prob)
            logit = safe_logit_from_prob(prob)
            include_hits, exclude_hits = keyword_rule_signal(
                text, include_keywords=include_keywords, exclude_keywords=exclude_keywords
            )
            dbg = {
                "token_matches": 0,
                "include_hits": include_hits,
                "exclude_hits": exclude_hits,
            }

        if prob >= accept_t:
            decision = "accept"
        elif prob <= reject_t:
            decision = "reject"
        else:
            decision = "review"

        append_jsonl(
            paths.scored_chunks,
            {
                "chunk_id": cid,
                "domain": args.domain,
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "text": text,
                "score": round(prob, 6),
                "logit": round(logit, 6),
                "decision": decision,
                "backend": backend,
                "token_matches": dbg["token_matches"],
                "include_hits": dbg["include_hits"],
                "exclude_hits": dbg["exclude_hits"],
                "scored_at": now_iso(),
            },
        )
        existing.add(cid)
        counts[decision] += 1
        new_scores += 1
        if args.max_new_scores is not None and new_scores >= args.max_new_scores:
            break

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "score",
            "backend": backend,
            "new_scores": new_scores,
            "accept": counts["accept"],
            "review": counts["review"],
            "reject": counts["reject"],
        },
    )
    print("Domain:", args.domain)
    print("Backend:", backend)
    print("New scored chunks:", new_scores)
    print("Accept:", counts["accept"])
    print("Review:", counts["review"])
    print("Reject:", counts["reject"])
    print("Scored chunk file:", paths.scored_chunks)
    return 0


def promote_accepts(args: argparse.Namespace) -> int:
    _, _, data_root = resolve_config_and_domain(args)
    paths = domain_paths(data_root, args.domain)
    paths.base.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(paths.scored_chunks)
    if not rows:
        raise ValueError(f"No scored rows found: {paths.scored_chunks}. Run `score` first.")

    seed_chunk_ids = load_jsonl_ids(paths.seed_chunks, "chunk_id")
    seed_urls = load_jsonl_ids(paths.seed_urls, "url")

    promoted_chunks = 0
    promoted_urls = 0
    for row in rows:
        if row.get("decision") != "accept":
            continue
        score = float(row.get("score", 0.0))
        if score < args.min_score:
            continue

        cid = row.get("chunk_id")
        if not isinstance(cid, str) or cid in seed_chunk_ids:
            continue

        append_jsonl(
            paths.seed_chunks,
            {
                "chunk_id": cid,
                "domain": args.domain,
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "score": score,
                "promoted_at": now_iso(),
                "source": "discriminator_accept",
            },
        )
        seed_chunk_ids.add(cid)
        promoted_chunks += 1

        raw_url = row.get("url", "")
        if isinstance(raw_url, str) and raw_url:
            u = canonicalize_url(raw_url)
            if u not in seed_urls:
                append_jsonl(
                    paths.seed_urls,
                    {
                        "url": u,
                        "domain": args.domain,
                        "source": "promoted_chunk_url",
                        "created_at": now_iso(),
                    },
                )
                seed_urls.add(u)
                promoted_urls += 1

        if args.max_promotions is not None and promoted_chunks >= args.max_promotions:
            break

    append_jsonl(
        paths.run_log,
        {
            "at": now_iso(),
            "domain": args.domain,
            "step": "promote",
            "promoted_chunks": promoted_chunks,
            "promoted_urls": promoted_urls,
            "seed_chunks_total": len(seed_chunk_ids),
            "seed_urls_total": len(seed_urls),
        },
    )
    print("Domain:", args.domain)
    print("Promoted chunks:", promoted_chunks)
    print("New promoted URLs:", promoted_urls)
    print("Seed chunk file:", paths.seed_chunks)
    print("Seed URL file:", paths.seed_urls)
    return 0


def run_iteration(args: argparse.Namespace) -> int:
    init_domain(args)
    discover_urls(args)
    ingest_candidates(args)
    bootstrap_labels(args)
    train_discriminator(args)
    score_candidates(args)
    promote_accepts(args)
    print("Iteration complete for domain:", args.domain)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed corpus discovery and enhancement pipeline (no QA generation)."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--domain", required=True)

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize domain directories and seed URLs.")

    discover = sub.add_parser("discover", help="Discover URLs by crawling from seed URLs.")
    discover.add_argument("--max-depth", type=int, default=None)
    discover.add_argument("--max-pages", type=int, default=None)
    discover.add_argument("--timeout", type=int, default=None)
    discover.add_argument("--delay", type=float, default=None)
    discover.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

    ingest = sub.add_parser("ingest", help="Fetch discovered URLs and write candidate chunks.")
    ingest.add_argument("--max-urls", type=int, default=None)
    ingest.add_argument("--timeout", type=int, default=None)
    ingest.add_argument("--delay", type=float, default=None)
    ingest.add_argument("--chunk-words", type=int, default=None)
    ingest.add_argument("--chunk-overlap", type=int, default=None)
    ingest.add_argument("--min-chars", type=int, default=None)
    ingest.add_argument("--max-chunks", type=int, default=None)
    ingest.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

    labels = sub.add_parser("bootstrap-labels", help="Create weak labels for first discriminator.")
    labels.add_argument("--max-pos", type=int, default=400)
    labels.add_argument("--max-neg", type=int, default=400)
    labels.add_argument("--min-include-hits", type=int, default=2)
    labels.add_argument("--min-exclude-hits", type=int, default=1)

    train = sub.add_parser("train", help="Train relevance discriminator (fastText or Naive Bayes).")
    train.add_argument(
        "--backend",
        choices=["fasttext", "naive_bayes"],
        default="fasttext",
        help="Discriminator backend. Use fasttext to match DeepSeek-style corpus mining.",
    )
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--min-df", type=int, default=2)
    train.add_argument("--max-vocab", type=int, default=25000)
    train.add_argument("--min-token-len", type=int, default=2)
    train.add_argument("--keyword-boost", type=float, default=0.4)
    train.add_argument("--ft-lr", type=float, default=0.8)
    train.add_argument("--ft-epoch", type=int, default=25)
    train.add_argument("--ft-word-ngrams", type=int, default=2)
    train.add_argument("--ft-dim", type=int, default=100)
    train.add_argument("--ft-minn", type=int, default=2)
    train.add_argument("--ft-maxn", type=int, default=5)
    train.add_argument("--ft-loss", choices=["softmax", "hs", "ova"], default="softmax")
    train.add_argument("--ft-thread", type=int, default=0)

    score = sub.add_parser("score", help="Score candidate chunks and assign decisions.")
    score.add_argument(
        "--backend",
        choices=["fasttext", "naive_bayes"],
        default=None,
        help="If omitted, auto-detects from trained model info.",
    )
    score.add_argument("--accept-threshold", type=float, default=0.82)
    score.add_argument("--reject-threshold", type=float, default=0.28)
    score.add_argument("--max-new-scores", type=int, default=None)

    promote = sub.add_parser("promote", help="Promote accepted chunks into the seed corpus.")
    promote.add_argument("--min-score", type=float, default=0.85)
    promote.add_argument("--max-promotions", type=int, default=None)

    iterate = sub.add_parser("iterate", help="Run one complete iteration end-to-end.")
    iterate.add_argument("--max-depth", type=int, default=None)
    iterate.add_argument("--max-pages", type=int, default=None)
    iterate.add_argument("--max-urls", type=int, default=None)
    iterate.add_argument("--timeout", type=int, default=None)
    iterate.add_argument("--delay", type=float, default=None)
    iterate.add_argument("--chunk-words", type=int, default=None)
    iterate.add_argument("--chunk-overlap", type=int, default=None)
    iterate.add_argument("--min-chars", type=int, default=None)
    iterate.add_argument("--max-chunks", type=int, default=None)
    iterate.add_argument("--max-pos", type=int, default=400)
    iterate.add_argument("--max-neg", type=int, default=400)
    iterate.add_argument("--min-include-hits", type=int, default=2)
    iterate.add_argument("--min-exclude-hits", type=int, default=1)
    iterate.add_argument("--backend", choices=["fasttext", "naive_bayes"], default="fasttext")
    iterate.add_argument("--alpha", type=float, default=1.0)
    iterate.add_argument("--min-df", type=int, default=2)
    iterate.add_argument("--max-vocab", type=int, default=25000)
    iterate.add_argument("--min-token-len", type=int, default=2)
    iterate.add_argument("--keyword-boost", type=float, default=0.4)
    iterate.add_argument("--ft-lr", type=float, default=0.8)
    iterate.add_argument("--ft-epoch", type=int, default=25)
    iterate.add_argument("--ft-word-ngrams", type=int, default=2)
    iterate.add_argument("--ft-dim", type=int, default=100)
    iterate.add_argument("--ft-minn", type=int, default=2)
    iterate.add_argument("--ft-maxn", type=int, default=5)
    iterate.add_argument("--ft-loss", choices=["softmax", "hs", "ova"], default="softmax")
    iterate.add_argument("--ft-thread", type=int, default=0)
    iterate.add_argument("--accept-threshold", type=float, default=0.82)
    iterate.add_argument("--reject-threshold", type=float, default=0.28)
    iterate.add_argument("--max-new-scores", type=int, default=None)
    iterate.add_argument("--min-score", type=float, default=0.85)
    iterate.add_argument("--max-promotions", type=int, default=None)
    iterate.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "init":
        return init_domain(args)
    if args.cmd == "discover":
        return discover_urls(args)
    if args.cmd == "ingest":
        return ingest_candidates(args)
    if args.cmd == "bootstrap-labels":
        return bootstrap_labels(args)
    if args.cmd == "train":
        return train_discriminator(args)
    if args.cmd == "score":
        return score_candidates(args)
    if args.cmd == "promote":
        return promote_accepts(args)
    if args.cmd == "iterate":
        return run_iteration(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
