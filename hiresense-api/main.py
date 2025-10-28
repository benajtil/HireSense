from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import io, re, chardet, base64, json

import pdfplumber
from docx import Document

# ---------- NLP / Skills ----------
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====== Setup ======
app = FastAPI(title="HireSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# spaCy
nlp = spacy.load("en_core_web_sm")

# Embeddings (semantic match)
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load canonical skills gazetteer (allow-list + aliases)
with open("skills_gazetteer.json", "r", encoding="utf-8") as f:
    CANON_SKILLS: Dict[str, List[str]] = json.load(f)

CANON_SET = set(CANON_SKILLS.keys())
ALIASES = {alias.lower(): canon for canon, als in CANON_SKILLS.items() for alias in als}
CANON_LIST = list(CANON_SET)

# Build PhraseMatcher patterns (canon + aliases), case-insensitive
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
def _mk_phrase(s: str): return nlp.make_doc(s.strip().lower())

all_phrases = {*(k.lower().strip() for k in CANON_SKILLS.keys())}
for als in CANON_SKILLS.values():
    for a in als:
        all_phrases.add(a.lower().strip())
matcher.add("SKILL", [_mk_phrase(p) for p in all_phrases if p])

SOFT_STOPWORDS = {
    "ability","accountability","accuracy","adaptability","communication","leadership",
    "ownership","responsibility","results oriented","teamwork","collaboration"
}

CLEAN_EDGES = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.+# -]+$")
NUMBER = re.compile(r"\b(\d+(\.\d+)?%?|\d{4})\b")

def _clean_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = CLEAN_EDGES.sub("", s)
    return re.sub(r"\s+", " ", s)

def _canonize(term: str) -> Optional[str]:
    t = _clean_token(term)
    if not t or t in SOFT_STOPWORDS:
        return None
    if t in CANON_SET:
        return t
    if t in ALIASES:
        return ALIASES[t]
    # conservative fuzzy fallback
    m = process.extractOne(t, CANON_LIST, scorer=fuzz.WRatio, score_cutoff=92)
    return m[0] if m else None

def extract_canonical_skills(text: str) -> set[str]:
    """Phrase-match against allow-list + aliases, then canonize; backstop with noun-chunks."""
    if not text:
        return set()

    doc = nlp(text)
    hits: set[str] = set()

    # 1) PhraseMatcher hits
    for _, start, end in matcher(doc):
        span = doc[start:end].text
        c = _canonize(span)
        if c:
            hits.add(c)

    # 2) Noun-chunk backstop (short chunks only)
    for ch in doc.noun_chunks:
        c = _canonize(ch.text)
        if c in CANON_SET:
            hits.add(c)

    return hits

def norm(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "")).strip().lower()

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return (len(sa & sb) / max(1, len(sa | sb))) if (sa or sb) else 0.0

def has_metrics(text: str) -> bool:
    return bool(NUMBER.search(text or ""))

# ---------- File extraction ----------
async def _pdf(bytes_: bytes) -> str:
    out = ""
    with pdfplumber.open(io.BytesIO(bytes_)) as pdf:
        for p in pdf.pages:
            out += "\n" + (p.extract_text() or "")
    return out.strip()

async def _docx(bytes_: bytes) -> str:
    with io.BytesIO(bytes_) as f:
        doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs).strip()

async def _txt(bytes_: bytes) -> str:
    guess = chardet.detect(bytes_) or {}
    enc = guess.get("encoding","utf-8") or "utf-8"
    try:
        return bytes_.decode(enc, errors="ignore")
    except Exception:
        return bytes_.decode("utf-8", errors="ignore")

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    raw = await file.read()

    if name.endswith(".pdf"):
        text = await _pdf(raw)
        preview_b64 = "data:application/pdf;base64," + base64.b64encode(raw).decode("ascii")
    elif name.endswith(".docx"):
        text = await _docx(raw)
        preview_b64 = None
    else:
        text = await _txt(raw)
        preview_b64 = None

    return {"text": text, "pdfPreview": preview_b64}

# ---------- Analyze ----------
class AnalyzeIn(BaseModel):
    resume_text: str
    job_text: str

class AnalyzeOut(BaseModel):
    fit_score: float
    breakdown: Dict[str, float]
    missing_skills: List[str]
    suggested_keywords: List[str]
    education: Dict[str, bool]
    sections: Dict[str, bool]
    notes: List[str]
    ats: Dict[str, bool]

SECTION_HEADERS = ["education","experience","work experience","skills","projects","certifications"]
DEGREE_TERMS = ["bachelor","bs","b.s.","ba","b.a.","bsc","b.sc","master","ms","m.s.","ma","m.a.","msc","m.sc","phd","doctorate","associate"]
MAJOR_TERMS = ["computer science","software engineering","information technology","computer engineering","information systems"]

def sections_map(text: str) -> Dict[str,bool]:
    t = norm(text)
    return {h: (h in t) for h in SECTION_HEADERS}

def education_signals(text: str) -> Dict[str,bool]:
    t = norm(text)
    return {
        "has_degree": any(k in t for k in DEGREE_TERMS),
        "has_major":  any(k in t for k in MAJOR_TERMS),
        "has_institution": bool(re.search(r"\b(university|college|institute|school)\b", t))
    }

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(inp: AnalyzeIn):
    resume_raw = inp.resume_text or ""
    job_raw    = inp.job_text or ""
    R, J = norm(resume_raw), norm(job_raw)

    # Semantic (embeddings)
    emb = embed.encode([R, J])
    semantic = float(cosine_similarity([emb[0]], [emb[1]])[0][0])  # 0..1

    # Canonical skills
    res_sk = extract_canonical_skills(resume_raw)
    jd_sk  = extract_canonical_skills(job_raw)
    missing = sorted(s for s in jd_sk if s not in res_sk)
    overlap = jaccard(list(res_sk), list(jd_sk))

    # Structure / Evidence
    secs = sections_map(resume_raw)
    edu  = education_signals(resume_raw)
    structure = sum([
        secs.get("education", False),
        secs.get("skills", False),
        secs.get("experience", False) or secs.get("work experience", False)
    ]) / 3.0
    evidence = 1.0 if has_metrics(resume_raw) else 0.0

    # Weighted score
    w_sem, w_skill, w_struct, w_evid = 0.55, 0.30, 0.10, 0.05
    score = w_sem*semantic + w_skill*overlap + w_struct*structure + w_evid*evidence
    score_pct = round(100*score, 1)

    # Suggestions (concrete + ATS friendly)
    notes: List[str] = []
    if missing:
        notes.append("Add truthful keywords in Skills/bullets: " + ", ".join(missing[:8]) + ".")
    if not evidence:
        notes.append("Quantify impact in bullets (%, counts, time).")
    if not edu["has_degree"]:
        notes.append("Add a degree line in Education (e.g., “B.S. in Computer Science, 2024”).")
    if not secs.get("skills", False):
        notes.append("Add a dedicated “Skills” section with your core stack.")

    # ATS heuristics
    ats = {
        "parseable": True,
        "headers_ok": bool(secs.get("education") and (secs.get("experience") or secs.get("work experience")) and secs.get("skills")),
        "contrast_ok": True,
    }

    breakdown = {
        "semantic": round(100*semantic, 1),
        "skills": round(100*overlap, 1),
        "structure": round(100*structure, 1),
        "evidence": 100.0 if evidence else 0.0,
    }

    return AnalyzeOut(
        fit_score=score_pct,
        breakdown=breakdown,
        missing_skills=missing,
        suggested_keywords=missing[:10],
        education=edu,
        sections=secs,
        notes=notes,
        ats=ats,
    )
