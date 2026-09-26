"""Name / address normalization (vectorized in polars; Python only for non-ASCII rows).

Output columns per record:
  name_norm   all name tokens after cleaning + abbreviation expansion
  name_core   name_norm minus legal suffixes / honorifics / filler (the distinctive part)
  name_alt    the part before "DBA" if present (else empty)
  addr_norm   address tokens after cleaning + abbreviation canonicalisation
  addr_nums   space-joined numeric tokens of the address (house / plot / zip numbers)
  name_loc    name_core minus tokens that also occur in the record's own address and country
              words ("Bordeaux Loisirs" at "..., Bordeaux" -> "loisirs"): the part of the name
              that is not a place
  street      street-name tokens: the address segment holding the house number (else the one
              with a street word), without numbers, street types and fillers ("pachn")
  house       first number of that street segment ("" if none): the house / plot number,
              independent of how a source reordered the address parts
"""
import json
import re

import polars as pl
from anyascii import anyascii

from config import WORK_DIR

INDIC_RE = re.compile(r"[ऀ-෿]")
NON_ASCII = r"[^\x00-\x7F]"
TRANSLIT_PATH = WORK_DIR / "translit_dict.json"

# ---------------------------------------------------------------- names
NAME_ABBR = {
    "pvt": "private", "prv": "private", "pvtltd": "private limited",
    "ltd": "limited", "ltda": "limited", "lt": "limited",
    "inc": "incorporated", "incorp": "incorporated", "corp": "corporation", "corpn": "corporation",
    "co": "company", "cos": "company", "coy": "company",
    "intl": "international", "int": "international", "natl": "national",
    "mfg": "manufacturing", "mfrs": "manufacturers", "svcs": "services", "svc": "services",
    "tech": "technologies", "technology": "technologies", "techs": "technologies",
    "assoc": "associates", "assn": "association", "bros": "brothers",
    "ent": "enterprises", "entp": "enterprises", "mgmt": "management", "dev": "development",
    "grp": "group", "hldgs": "holdings", "inds": "industries", "ind": "industries",
    "shri": "sri", "shree": "sri", "sree": "sri", "shre": "sri",
    "n": "and", "et": "and",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc",
    "pllc", "pc", "pa", "the", "and", "of", "m", "s", "ms", "dba", "a", "an", "sri",
    # French legal forms (test-only country; keep the list generic). "societe", "ste" and "cie"
    # are kept: in names like "Bordeaux Societe" they are the only non-place word
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "scop", "sca", "gie",
    "ets", "etablissements", "de", "la", "le", "les", "du", "des", "et", "l", "d",
    # other generic legal words
    "gmbh", "ag", "bv", "nv", "pty", "public",
}

# ---------------------------------------------------------------- addresses
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
US_STATES_MULTI = {  # two-word state names -> abbreviation (applied as regex before tokenizing)
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "district of columbia": "dc",
}
IN_STATES = {  # Indian state names -> short code (so "DL"/"Delhi"/"दिल्ली" all agree)
    "delhi": "dl", "maharashtra": "mh", "karnataka": "ka", "haryana": "hr", "gujarat": "gj",
    "rajasthan": "rj", "kerala": "kl", "punjab": "pb", "bihar": "br", "odisha": "od",
    "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "ua": "uk",
    "uttaranchal": "uk", "assam": "as", "goa": "ga", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "telangana": "ts", "tg": "ts", "manipur": "mn", "meghalaya": "ml",
    "tripura": "tr", "nagaland": "nl", "mizoram": "mz", "sikkim": "sk",
    # single-token forms produced by transliterating native-script state names
    "bengal": "wb", "westbengal": "wb", "tamil": "tn", "tamilnadu": "tn", "andhra": "ap",
    "uttarpradesh": "up", "madhyapradesh": "mp", "himachal": "hp", "kashmir": "jk",
}
IN_STATES_MULTI = {
    "uttar pradesh": "up", "west bengal": "wb", "tamil nadu": "tn", "andhra pradesh": "ap",
    "madhya pradesh": "mp", "himachal pradesh": "hp", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "arunachal pradesh": "ar", "bangalore": "bengaluru",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
}
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "stree": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "crt": "ct",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "bld": "blvd", "bvd": "blvd",
    "place": "pl", "circle": "cir", "circ": "cir", "highway": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pky": "pkwy", "trail": "trl", "terrace": "ter", "square": "sq",
    "mount": "mt", "mountain": "mtn", "point": "pt", "heights": "hts", "center": "ctr",
    "centre": "ctr", "expressway": "expy", "freeway": "fwy", "junction": "jct",
    "crossing": "xing", "cross": "crs", "main": "mn",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "fort": "ft", "ftt": "ft",
    "nagar": "ngr", "colony": "col", "sector": "sec", "sect": "sec", "layout": "lyt",
    "extension": "extn", "ext": "extn", "phase": "ph", "block": "blk", "building": "bldg",
    # floor -> "fl" (not "flr") so the state code "FL" and "Florida" stay the same token
    "apartment": "apt", "apartments": "apt", "apts": "apt", "floor": "fl", "flr": "fl",
    "opposite": "opp", "near": "nr", "behind": "bh", "district": "dist", "distt": "dist",
    "taluk": "tq", "taluka": "tq", "tehsil": "teh", "village": "vill", "vil": "vill",
    "post": "po", "industrial": "indl", "ind": "indl", "estate": "est", "complex": "cplx",
    # French
    "r": "rue", "chemin": "ch", "chem": "ch", "route": "rte", "allee": "all", "impasse": "imp",
    "faubourg": "fbg", "fg": "fbg", "quai": "qu", "cours": "crs", "residence": "res",
    **ORDINAL_WORDS,
}
COUNTRY_WORDS = ["france", "india", "usa", "us", "america", "bharat"]
# street types / fillers removed from the street column (canonical forms after ADDR_ABBR)
STREET_WORDS = {
    "rue", "ave", "blvd", "ch", "rte", "all", "imp", "fbg", "qu", "crs", "res", "pl", "st", "rd",
    "ln", "dr", "ct", "cir", "hwy", "pkwy", "trl", "ter", "sq", "expy", "fwy", "way", "marg",
    "plot", "sec", "blk", "bldg", "fl", "nr", "opp", "bh", "ph", "extn", "ngr", "col",
    "de", "du", "des", "la", "le", "les", "l", "d", "n", "s", "e", "w", "ne", "nw", "se", "sw",
}
STREET_HINT = (r"\b(rue|r|avenue|av|bd|boulevard|chemin|impasse|allee|route|quai|place|street|st"
               r"|road|rd|lane|ln|drive|dr|marg|nagar|sector|block)\b")
ADDR_DROP = {"null", "none", "na", "nan", "no", "ndeg", "number", "num", "nos", "h", "hn", "hno", "door",
             "dno", "box", "unit", "suite", "ste", "apt", "house", "shop", "flat", "at", "and",
             "of", "the", "po", "pin", "pincode", "india", "usa", "us", "france", "bis"}


def _load_translit() -> dict:
    if TRANSLIT_PATH.exists():
        with open(TRANSLIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


_TL = _load_translit()
NAME_TL, ADDR_TL = {}, {}


def use_translit(fold: int | None = None):
    """Select the full dictionary (fold=None) or the one learned without S1-fold `fold`."""
    global NAME_TL, ADDR_TL
    d = _TL if fold is None else _TL["folds"][fold]
    NAME_TL, ADDR_TL = d.get("name", {}), d.get("addr", {})


def n_translit_folds() -> int:
    return len(_TL.get("folds", []))


def reload_translit():
    global _TL
    _TL = _load_translit()
    use_translit()


use_translit()


def ascii_key(tok: str) -> str:
    """Lowercased alnum-only ASCII form of one raw token (dictionary key for Indic tokens)."""
    return re.sub(r"[^a-z0-9]", "", anyascii(tok).lower())


def _translit(text: str, table: dict) -> str:
    if not INDIC_RE.search(text):
        return anyascii(text)
    out = []
    for tok in text.split():
        if INDIC_RE.search(tok):
            k = ascii_key(tok)
            out.append(table.get(k, k))
        else:
            out.append(anyascii(tok))
    return " ".join(out)


def translit_name(s: str) -> str:
    return _translit(s, NAME_TL)


def translit_addr(s: str) -> str:
    # addresses keep commas; transliterate each comma part so separators survive
    return ",".join(_translit(p, ADDR_TL) for p in s.split(","))


def _ascii_expr(col: str, fn) -> pl.Expr:
    c = pl.col(col).fill_null("")
    return (pl.when(c.str.contains(NON_ASCII))
            .then(c.map_elements(fn, return_dtype=pl.String, skip_nulls=True))
            .otherwise(c))


def _map_tokens(expr: pl.Expr, table: dict, drop: set | None = None) -> pl.Expr:
    toks = expr.str.split(" ").list.eval(pl.element().replace(table)).list.join(" ")
    toks = toks.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    if drop:
        toks = toks.list.eval(pl.element().filter(~pl.element().is_in(list(drop))))
    return toks.list.join(" ")


def name_exprs(col: str = "business_name") -> list[pl.Expr]:
    s = _ascii_expr(col, translit_name).str.to_lowercase()
    s = (s.str.replace_all(r"\bwww\.", " ")
          .str.replace_all(r"\.(com|net|org|in|co|biz|info|fr|us|io)\b", " ")
          .str.replace_all(r"&", " and ")
          .str.replace_all(r"#\s*\d+", " ")          # store numbers "#38562"
          .str.replace_all(r"\bm/s\b", " ")
          .str.replace_all(r"\bd\s*/?\s*b\s*/?\s*a\b|\ba\s*/?\s*k\s*/?\s*a\b|\bt\s*/\s*a\b"
                           r"|\bf\s*/?\s*k\s*/?\s*a\b|\btrading as\b|\bdoing business as\b"
                           r"|\bformerly(?: known as)?\b|\balso known as\b", " dba ")
          .str.replace_all(r"'s\b", "s")
          .str.replace_all(r"[^a-z0-9]+", " ")
          .str.strip_chars())
    norm = _map_tokens(s, NAME_ABBR)
    after_dba = norm.str.replace(r"^.*\bdba\b", "").str.strip_chars()
    before_dba = pl.when(norm.str.contains(r"\bdba\b")).then(
        norm.str.replace(r"\bdba\b.*$", "").str.strip_chars()).otherwise(pl.lit(""))
    core = _map_tokens(pl.when(after_dba == "").then(norm).otherwise(after_dba), {}, LEGAL)
    # collapse repeated tokens ("empire empire liberty")
    core = core.str.split(" ").list.unique(maintain_order=True).list.join(" ")
    return [norm.alias("name_norm"), core.alias("name_core"),
            _map_tokens(before_dba, {}, LEGAL).alias("name_alt")]


def _addr_tokens(s: pl.Expr) -> pl.Expr:
    s = (s.str.replace_all(r"\b[cswd]\s*/\s*o\b", " ")             # c/o, s/o, w/o, d/o
          .str.replace_all(r"(\d+)(st|nd|rd|th)\b", "$1")         # 7th / 7nd / 7rd -> 7 ("45 St" kept)
          .str.replace_all(r"(\d)([a-z])", "$1 $2")
          .str.replace_all(r"([a-z])(\d)", "$1 $2")
          .str.replace_all(r"[^a-z0-9]+", " ")
          .str.strip_chars())
    norm = _map_tokens(s, {**ADDR_ABBR, **US_STATES, **IN_STATES}, ADDR_DROP)
    return norm.str.replace_all(r"\b0+(\d)", "$1")                   # 001555 -> 1555


def addr_exprs(col: str = "business_address") -> list[pl.Expr]:
    s = _ascii_expr(col, translit_addr).str.to_lowercase()
    for k, v in {**US_STATES_MULTI, **IN_STATES_MULTI}.items():
        s = s.str.replace_all(rf"\b{k}\b", v)
    norm = _addr_tokens(s)
    nums = norm.str.extract_all(r"\b\d+\b").list.join(" ")
    # street segment: sources reorder the comma parts ("Gironde, BORDEAUX, 18 R ..."), so take
    # the part with the house number, else the one naming a street type
    segs = s.str.split(",")
    seg = pl.coalesce(segs.list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first(),
                      segs.list.eval(pl.element().filter(pl.element().str.contains(STREET_HINT))).list.first(),
                      pl.lit(""))
    seg_tokens = _addr_tokens(seg)
    street = (seg_tokens.str.extract_all(r"\S+")
              .list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")
                                             & ~pl.element().is_in(list(STREET_WORDS))))
              .list.join(" "))
    # house number: first number of the street segment. Sources reorder the comma parts, so the
    # first number of the whole address is often a plot / floor / sector number on one side only
    house = seg_tokens.str.extract(r"\b(\d+)\b").fill_null("")
    return [norm.alias("addr_norm"), nums.alias("addr_nums"), street.alias("street"),
            house.alias("house")]


def normalize(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    loc = pl.col("addr_norm").str.extract_all(r"\S+").list.concat(pl.lit(COUNTRY_WORDS))
    name_loc = (pl.col("name_core").str.extract_all(r"\S+").list.set_difference(loc)
                .list.join(" "))
    return df.lazy().with_columns(*name_exprs(), *addr_exprs()).with_columns(name_loc=name_loc)
