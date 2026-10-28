import re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
import os
import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from duckduckgo_search import DDGS
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
phase1_env = os.getenv("PHASE1_DIR", "models/phase1_roberta_fulltune/best")
if not Path(phase1_env).exists():
    if Path("models/phase1_roberta_fulltune").exists():
        PHASE1_DIR = "models/phase1_roberta_fulltune"
    else:
        PHASE1_DIR = "roberta-base"
else:
    PHASE1_DIR = phase1_env

phase2_env = os.getenv("PHASE2_HEAD", "models/phase2_fusion_head/fusion_head.pt")
if not Path(phase2_env).exists():
    if Path("fusion_head.pt").exists():
        PHASE2_HEAD = "fusion_head.pt"
    else:
        PHASE2_HEAD = phase2_env
else:
    PHASE2_HEAD = phase2_env

app = FastAPI(title="Fake News Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Models ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(PHASE1_DIR, use_fast=False)
roberta = AutoModelForSequenceClassification.from_pretrained(
    PHASE1_DIR, num_labels=2, output_attentions=True
).to(device)
roberta.eval()


class FusionHead(nn.Module):
    def __init__(self, semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + stat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, semantic, stat):
        return self.net(torch.cat([semantic, stat], dim=-1))


fusion_head = None
stat_mean = np.array([3.6115989685058594, 110.91300201416016, -3.6115989685058594], dtype=np.float32)
stat_std = np.array([0.3903583288192749, 1401.9859619140625, 0.3903583288192749], dtype=np.float32)

if Path(PHASE2_HEAD).exists():
    try:
        ckpt = torch.load(PHASE2_HEAD, map_location=device, weights_only=False)
        hidden = ckpt.get("hidden", 256)
        if "stat_mean" in ckpt:
            stat_mean = np.array(ckpt["stat_mean"], dtype=np.float32)
        if "stat_std" in ckpt:
            stat_std = np.array(ckpt["stat_std"], dtype=np.float32)
        fusion_head = FusionHead(hidden=hidden).to(device)
        fusion_head.load_state_dict(ckpt["head_state"])
        fusion_head.eval()
        print(f"[TruthShield] Phase 2 Fusion Head loaded successfully (hidden={hidden}).")
    except Exception as e:
        print(f"[TruthShield] Warning: Could not load Phase 2 head: {e}")

gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_mod.eval()

# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_word(word: str) -> str:
    word = word.replace("\u00e2\u0080\u0099", "'")
    word = word.replace("\u00e2\u0080\u009c", '"')
    word = word.replace("\u00e2\u0080\u009d", '"')
    word = word.replace("\u010c\u201c", '"')
    word = word.replace("\u010c\u2122", "'")
    word = word.replace("\u00e2\u0122\u017e", '"')
    word = word.replace("\u00e2\u0122\u00be", '"')
    word = word.replace("\u00e2\u0122\u0080", "...")
    word = word.replace("\u00e2\u0122\u0081", "-")
    word = word.replace("\u00c4\u0141", "g")
    word = word.replace("\u00c4\u00b1", "i")
    word = re.sub(r"[^\x00-\x7F]+", "", word)
    word = re.sub(r" +", " ", word).strip()
    return word


def score_to_tier(score: float) -> int:
    if score > 0.92:
        return 4
    elif score > 0.78:
        return 3
    elif score > 0.62:
        return 2
    elif score > 0.48:
        return 1
    return 0


def build_token_highlights(attentions, enc):
    input_ids = enc["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    real_len = sum(enc["attention_mask"][0].cpu().tolist())

    seq_len = attentions[0][0].shape[-1]
    rollout = torch.eye(seq_len).to(attentions[0][0].device)
    for layer_attn in attentions:
        avg_heads = layer_attn[0].mean(dim=0)
        aug = avg_heads + torch.eye(seq_len).to(avg_heads.device)
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)

    token_importance = rollout[0].cpu().numpy()
    tokens = tokens[1:real_len - 1]
    scores = token_importance[1:real_len - 1]

    ranks = np.argsort(np.argsort(scores))
    scores = ranks / (len(ranks) - 1 + 1e-8)

    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        score = float(scores[i])
        word = tok.replace("\u0120", "").replace("\u2581", "")

        while (
            i + 1 < len(tokens)
            and not tokens[i + 1].startswith("\u0120")
            and not tokens[i + 1].startswith("\u2581")
        ):
            i += 1
            word += tokens[i].replace("\u0120", "").replace("\u2581", "")
            score = max(score, float(scores[i]))

        word = clean_word(word)
        if word:
            result.append({
                "word": word,
                "tier": score_to_tier(score),
                "leading_space": tok.startswith("\u0120") or tok.startswith("\u2581"),
            })
        i += 1

    return result


@torch.no_grad()
def compute_stat_features(text: str):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text) if text.strip() else [""]
    lengths = [len(s.split()) for s in sents]
    sent_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    loss = 0.0
    if len(text.split()) >= 2:
        enc = gpt2_tok(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)
        loss = gpt2_mod(**enc, labels=enc.input_ids).loss.item()
    return np.array([[loss, sent_var, -loss]], dtype=np.float32)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return "Hello project is running"


class AnalyzeRequest(BaseModel):
    text: str
    model: str = "phase2"
    # 'model' = ML only | 'agent' = LLM agent only | 'both' = full pipeline
    agent_mode: str = "both"


@app.post("/analyze")
@torch.no_grad()
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    enc = tokenizer(
        text, truncation=True, max_length=512, padding=True, return_tensors="pt"
    ).to(device)

    if req.model == "phase2" and fusion_head is not None:
        roberta_out = roberta.roberta(**enc, output_attentions=True)
        cls_emb = roberta_out.last_hidden_state[:, 0, :]
        attentions = roberta_out.attentions
        stat_raw = compute_stat_features(text)
        stat_norm = (stat_raw - stat_mean) / (stat_std + 1e-8)
        logits = fusion_head(
            cls_emb.to(device),
            torch.tensor(stat_norm, dtype=torch.float32).to(device),
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    else:
        out = roberta(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        attentions = out.attentions

    pred = int(probs.argmax())
    label = "REAL" if pred == 0 else "FAKE"
    tokens = build_token_highlights(attentions, enc)

    return {
        "label": label,
        "prob_real": round(float(probs[0]), 4),
        "prob_fake": round(float(probs[1]), 4),
        "tokens": tokens,
    }


@app.post("/analyze_stream")
async def analyze_stream(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    def run_local_ml():
        with torch.no_grad():
            enc = tokenizer(
                text, truncation=True, max_length=512, padding=True, return_tensors="pt"
            ).to(device)

            if req.model == "phase2" and fusion_head is not None:
                roberta_out = roberta.roberta(**enc, output_attentions=True)
                cls_emb = roberta_out.last_hidden_state[:, 0, :]
                attentions = roberta_out.attentions
                stat_raw = compute_stat_features(text)
                stat_norm = (stat_raw - stat_mean) / (stat_std + 1e-8)
                logits = fusion_head(
                    cls_emb.to(device),
                    torch.tensor(stat_norm, dtype=torch.float32).to(device),
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            else:
                out = roberta(**enc)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
                attentions = out.attentions

            pred = int(probs.argmax())
            label = "REAL" if pred == 0 else "FAKE"
            tokens = build_token_highlights(attentions, enc)
            return label, probs, tokens

    def search_web_claims(claims):
        results = ""
        tinyfish_key = os.getenv("TINYFISH_API_KEY", "").strip()

        # 1. Primary: TinyFish Web Search API
        if tinyfish_key:
            try:
                headers = {"X-API-Key": tinyfish_key}
                resp = requests.get(
                    "https://api.search.tinyfish.ai",
                    params={
                        "query": claims.strip(),
                        "domain_type": "web",
                        "purpose": "Verify truthfulness of factual statements and news claims",
                    },
                    headers=headers,
                    timeout=12,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_items = data.get("results", [])
                    top_urls = []
                    for item in raw_items[:4]:
                        title = item.get("title", "").strip()
                        snippet = item.get("snippet", "").strip()
                        url = item.get("url", "").strip()
                        site = item.get("site_name", "").strip()
                        date = item.get("date", "").strip()
                        if title:
                            d_str = f" [{date}]" if date else ""
                            results += f"Source: {site}{d_str}\nTitle: {title}\n"
                            if snippet:
                                results += f"Snippet: {snippet}\n"
                            if url:
                                results += f"URL: {url}\n"
                            results += "\n"
                        if url and len(top_urls) < 1 and not snippet:
                            top_urls.append(url)

                    # If snippets are empty/short and we have a top URL, extract markdown text using TinyFish Fetch API
                    if top_urls:
                        try:
                            fetch_resp = requests.post(
                                "https://api.fetch.tinyfish.ai",
                                json={
                                    "urls": top_urls,
                                    "format": "markdown",
                                    "ttl": 3600,
                                    "per_url_timeout_ms": 10000,
                                },
                                headers={
                                    "X-API-Key": tinyfish_key,
                                    "Content-Type": "application/json",
                                },
                                timeout=12,
                            )
                            if fetch_resp.status_code == 200:
                                fdata = fetch_resp.json()
                                for fitem in fdata.get("results", []):
                                    ftext = (fitem.get("text") or "").strip()[:800]
                                    if ftext:
                                        results += f"Direct Source Evidence ({fitem.get('title', '')}):\n{ftext}\n\n"
                        except Exception as fe:
                            print(f"TinyFish fetch note: {fe}")

                    if results.strip():
                        return results.strip()
            except Exception as e:
                print(f"TinyFish search note: {e}")

        # 2. Fallback: DuckDuckGo news / text search
        try:
            with DDGS() as ddgs:
                news_items = list(ddgs.news(claims.strip(), max_results=3))
                if news_items:
                    for item in news_items:
                        results += f"Title: {item.get('title', '')}\nSnippet: {item.get('body', '')}\nSource: {item.get('source', '')}\n\n"
                    if results:
                        return results
                # Fallback to general ddgs text search if no news
                text_items = list(ddgs.text(claims.strip(), max_results=3))
                if text_items:
                    for item in text_items:
                        results += f"Title: {item.get('title', '')}\nSnippet: {item.get('body', '')}\n\n"
                    if results:
                        return results
        except Exception as e:
            print(f"DDGS search note: {e}")

        # 3. Fallback: GNews if key is present
        gnews_key = os.getenv("GNEWS_API_KEY")
        if gnews_key and gnews_key != "your_gnews_key":
            try:
                query = urllib.parse.quote(claims.strip())
                url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=3&apikey={gnews_key}"
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    for article in data.get("articles", []):
                        results += f"Title: {article['title']}\nSnippet: {article['description']}\n\n"
                    if results:
                        return results
            except Exception as e:
                print(f"GNews error: {e}")

        return results if results else "No strict verifications found."

    agent_mode = req.agent_mode  # 'model' | 'agent' | 'both'

    async def event_generator():
        # ── MODE: model only ────────────────────────────────────────────────────
        if agent_mode == "model":
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"
            return

        # Check API keys for LLM-based modes (Groq primary, NVIDIA secondary)
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        if nvidia_key:
            nvidia_key = nvidia_key.replace("Bearer ", "").strip()

        has_groq = bool(AsyncGroq and groq_key and groq_key != "your_groq_api_key")
        has_nvidia = bool(nvidia_key and nvidia_key != "your_gemma_api_key" and nvidia_key != "your_nvidia_api_key")

        if not has_groq and not has_nvidia:
            yield json.dumps({"type": "status", "message": "Notice: LLM API key not set in .env. Running high-precision local ML model..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"
            conf = max(probs[0], probs[1])
            msg = f"Analysis based on custom RoBERTa + GPT-2 statistical features (Model Confidence: {conf:.1%}, Verdict: {label}).\nTo enable live web search & LLM synthesis, set GROQ_API_KEY in backend/.env."
            yield json.dumps({"type": "llm_chunk", "content": msg}) + "\n\n"
            return

        # Initialize Groq client or NVIDIA client
        groq_client = None
        nvidia_client = None
        extract_prompt = f"Extract 2 to 4 crucial search keywords identifying the core claim, person, or event from this text to use as a web search query. Output ONLY the keywords separated by spaces. DO NOT use quotes, bullet points, or extra text:\n\n{text[:1500]}"
        claims = text[:80]

        if has_groq:
            groq_client = AsyncGroq(api_key=groq_key)
            yield json.dumps({"type": "status", "message": f"Extracting search parameters via {groq_model.split('/')[-1]} on Groq..."}) + "\n\n"
            try:
                extract_resp = await groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": extract_prompt}],
                    temperature=0.2,
                    max_tokens=60,
                )
                if extract_resp.choices and extract_resp.choices[0].message.content:
                    claims = extract_resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"Groq extract note: {e}")
                claims = text[:80]
        elif has_nvidia:
            nvidia_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
            yield json.dumps({"type": "status", "message": f"Extracting search parameters via {nvidia_model.split('/')[-1]}..."}) + "\n\n"
            try:
                nvidia_client = ChatNVIDIA(
                    model=nvidia_model,
                    api_key=nvidia_key,
                    temperature=0.7,
                    top_p=0.9,
                    max_completion_tokens=2048,
                )
                claims_res = await asyncio.to_thread(nvidia_client.invoke, [{"role": "user", "content": extract_prompt}])
                claims = str(claims_res.content).strip()
            except Exception as e:
                claims = text[:80]

        tinyfish_key = os.getenv("TINYFISH_API_KEY", "").strip()
        search_desc = "TinyFish Web Search" if tinyfish_key else "Live Web Engine"
        yield json.dumps({"type": "status", "message": f"Verifying facts against live web data via {search_desc}..."}) + "\n\n"
        web_context = await asyncio.to_thread(search_web_claims, claims)

        # ── MODE: both — also run ML scoring ───────────────────────────────────
        label, probs, tokens_result = None, None, []
        if agent_mode == "both":
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)

        yield json.dumps({"type": "status", "message": "Synthesizing Threat Intelligence..."}) + "\n\n"

        # Emit verdict + token highlights only when ML ran
        if agent_mode == "both" and probs is not None:
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"

        # Build the LLM final prompt
        web_has_hits = (
            "No strict verifications found" not in web_context
            and "Search failed" not in web_context
        )

        if agent_mode == "both" and probs is not None:
            ml_conf = max(probs[0], probs[1])
            ml_section = f"""## ML STRUCTURAL ANALYSIS (40% weight)
- Prediction: {label} | Confidence: {ml_conf:.2%}
- Note: Trained on journalistic articles (200-1000+ words). Short factual phrases are out-of-distribution.

## REASONING RULES
1. General facts, common knowledge, physical reality, and basic truths (e.g. "a car runs on 4 wheels", scientific laws, definitions) are ALWAYS evaluated for physical truth. If the statement is factually true in reality, verdict is REAL.
2. Web evidence CONFIRMS claims → lean REAL, use ML as a confidence booster.
3. Web evidence CONTRADICTS claims → lean FAKE regardless of ML output.
4. If the text is a short factual statement (fewer than 30 words), note the factual accuracy in reality while explaining that journalistic ML models are calibrated for long news articles.
5. Both signals agree → high-confidence verdict in that direction."""
            weight_note = "- Factual Verification & Web Evidence: 60% weight — PRIMARY signal.\n- Custom Trained ML Model (RoBERTa + Fusion Head): 40% weight — SECONDARY signal."
        else:
            ml_section = "## NOTE\nNo ML model output available — base your verdict on factual accuracy and web evidence."
            weight_note = "- Factual Verification & Live Evidence: 100% weight — sole signal for this analysis."

        final_prompt = f"""You are VIGIL-AI, an advanced threat intelligence system determining if a statement or news article is REAL or FAKE.

## SCORING WEIGHTS
{weight_note}

## TARGET TEXT
\"{text[:1000]}\"

## LIVE WEB EVIDENCE
{"CORROBORATING ARTICLES FOUND:" if web_has_hits else "NO CORROBORATING ARTICLES FOUND in live news sources (general knowledge verification applies)."}
{web_context}

{ml_section}

Write 2-3 sentences of reasoned analysis, then conclude:
**VERDICT: [REAL / FAKE / INCONCLUSIVE]** — one decisive bottom-line sentence.
"""

        if has_groq and groq_client is not None:
            try:
                stream = await groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.6,
                    max_tokens=2048,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta:
                        if hasattr(delta, "reasoning") and delta.reasoning:
                            yield json.dumps({"type": "llm_chunk", "reasoning": delta.reasoning}) + "\n\n"
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                            yield json.dumps({"type": "llm_chunk", "reasoning": delta.reasoning_content}) + "\n\n"
                        if delta.content:
                            yield json.dumps({"type": "llm_chunk", "content": delta.content}) + "\n\n"
            except Exception as e:
                yield json.dumps({"type": "llm_chunk", "content": f"\n\n[Groq Inference Error: {str(e)}]"}) + "\n\n"
        elif has_nvidia and nvidia_client is not None:
            try:
                async for chunk in nvidia_client.astream([{"role": "user", "content": final_prompt}]):
                    if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                        reasoning = chunk.additional_kwargs["reasoning_content"]
                        if reasoning:
                            yield json.dumps({"type": "llm_chunk", "reasoning": reasoning}) + "\n\n"
                    if chunk.content:
                        yield json.dumps({"type": "llm_chunk", "content": chunk.content}) + "\n\n"
            except Exception as e:
                yield json.dumps({"type": "llm_chunk", "content": f"\n\n[NVIDIA Inference Error: {str(e)}]"}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}