"""Shared helpers: FMP client with caching, universe, paths."""
import json, os, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

START, END = "2018-01-01", "2026-10-16"
BENCH = "XBI"

# Listed biotech / specialty pharma universe (alive in 2026; see README for the survivorship caveat).
UNIVERSE = sorted(set("""
VRTX REGN GILD AMGN BIIB MRNA ALNY BMRN INCY NBIX SRPT UTHR EXEL IONS HALO ARGX ACAD INSM RARE APLS CRNX BPMC IMVT MDGL AXSM KRYS PTCT CYTK VKTX RYTM ARWR DNLI ROIV PCVX SMMT NUVL IDYA RVMD
ILMN JAZZ BNTX NVAX ITCI TGTX LEGN BBIO MRUS ADMA CORT RNA IRON VERA ANAB KURA XENE PRAX ARVN RCKT TVTX FOLD NTLA CRSP BEAM EDIT SAGE BLUE ALKS ACLX GERN IOVA AGIO MNKD ABCL
ABBV PFE MRK LLY BMY JNJ
""".split()))

def _key():
    k = os.environ.get("FMP_API_KEY")
    if not k:
        env = Path(os.environ.get("FMP_ENV_FILE", "")).expanduser()
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("FMP_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise SystemExit("Set FMP_API_KEY (or FMP_ENV_FILE pointing at a .env that has it).")
    return k

def fmp(path, **q):
    """GET a Financial Modeling Prep endpoint, retrying on rate limits. Returns parsed JSON or None."""
    q["apikey"] = _key()
    url = "https://financialmodelingprep.com" + path + "?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1)); continue
            return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None

def cached(path: Path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    if data is not None:
        path.write_text(json.dumps(data))
    return data
