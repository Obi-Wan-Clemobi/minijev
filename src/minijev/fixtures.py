"""Fixed example requests and datasets' option sets, and a certificate-checked download."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

DATA = Path(os.environ.get("MINIJEV_DATA_DIR", Path.cwd() / "data"))

GDPR_REVISION = 1363040264  # the revision pinned by TypeSafe's parallel-questions cookbook


GDPR_QUESTIONS = {  # verbatim from https://docs.typesafe.ai/cookbooks/parallel_questions
    "breach_72h": {"type": "noul", "instructions": "Must a personal data breach be reported to the supervisory authority within 72 hours?"},
    "applies_non_eu": {"type": "noul", "instructions": "Does the regulation apply to organisations established outside the EU that offer goods or services to people in the EU?"},
    "dpo_all_orgs": {"type": "noul", "instructions": "Must every organisation appoint a Data Protection Officer, regardless of what data it processes?"},
    "pre_ticked_consent": {"type": "noul", "instructions": "Can valid consent be obtained through pre-ticked boxes or inactivity?"},
    "right_erasure": {"type": "noul", "instructions": "Does the regulation grant individuals a right to erasure of their personal data?"},
    "data_portability": {"type": "noul", "instructions": "Does the regulation include a right to data portability?"},
    "us_federal_law": {"type": "noul", "instructions": "Is the GDPR a United States federal law?"},
    "criminal_penalties": {"type": "noul", "instructions": "Does the GDPR itself impose criminal penalties such as imprisonment?"},
    "instrument_type": {"type": "choice", "instructions": "What kind of EU legal instrument is the GDPR?", "criteria": {
        "Regulation": "Directly binding law in all member states, no national implementation needed.",
        "Directive": "Sets goals that member states implement through national law.",
        "Treaty": "An international treaty between states.",
        "Recommendation": "Non-binding guidance."}},
    "max_fine": {"type": "choice", "instructions": "What is the maximum administrative fine for the most serious infringements?", "criteria": {
        "TwentyM_or_4pct": "Up to EUR 20 million or 4% of annual worldwide turnover, whichever is greater.",
        "TenM_or_2pct": "Up to EUR 10 million or 2% of annual worldwide turnover, whichever is greater.",
        "FixedCap": "A fixed amount not tied to turnover.",
        "NoFines": "The GDPR provides no administrative fines."}},
    "individual_rights": {"type": "score", "instructions": "How strong are the rights the GDPR grants to individuals over their data?", "criteria": [
        "None: individuals get no rights over their data.",
        "Weak: a right to be informed, but little control.",
        "Moderate: access and correction rights, but limited means to act on them.",
        "Strong: access, erasure, portability, and objection rights, with enforcement behind them."]},
    "penalty_severity": {"type": "score", "instructions": "How severe are the penalties the GDPR provides for non-compliance?", "criteria": [
        "None: no penalties of any kind.",
        "Symbolic: small fixed fines unlikely to change behavior.",
        "Substantial: fines large enough to matter to most companies.",
        "Severe: fines scaled to global revenue, material even to the largest companies."]},
    "compliance_burden": {"type": "score", "instructions": "How heavy is the compliance burden the GDPR places on organisations?", "criteria": [
        "Negligible: no meaningful obligations.",
        "Light: a few notices and disclosures.",
        "Moderate: documented processes and some dedicated roles for larger processors.",
        "Heavy: records, impact assessments, officers, and breach procedures for many organisations.",
        "Extreme: obligations so demanding that ordinary organisations cannot fully comply."]},
}


# Jev's batched means from the same cookbook (jev-1.12, full ~54k-character article).
GDPR_JEV = {"breach_72h": 0.804, "applies_non_eu": 0.990, "dpo_all_orgs": 0.030, "pre_ticked_consent": 0.040,
            "right_erasure": 0.990, "data_portability": 0.990, "us_federal_law": 0.010, "criminal_penalties": 0.108,
            "instrument_type": 1.000, "max_fine": 1.000, "individual_rights": 1.000, "penalty_severity": 1.000,
            "compliance_burden": 0.750}


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "minijev-poc/0.1"})
        # Certificates are checked. Behind a proxy that re-signs TLS, point SSL_CERT_FILE at its CA bundle.
        # MINIJEV_INSECURE_SSL=1 turns the check off; a download made that way could be tampered with.
        ssl_context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        if os.environ.get("MINIJEV_INSECURE_SSL") == "1":
            print(f"WARNING: certificate checks are off (MINIJEV_INSECURE_SSL=1) for {url}", flush=True)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ssl_context) as r:
            path.write_bytes(r.read())
    return path


def gdpr_text() -> str:
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           f"&prop=extracts&explaintext=1&revids={GDPR_REVISION}")
    pages = json.loads(fetch(url, DATA / f"gdpr_{GDPR_REVISION}.json").read_text())["query"]["pages"]
    return next(iter(pages.values()))["extract"]


def gdpr_state(engine: Engine, n_tokens: int | None) -> dict:
    """The cookbook's state shape, with the article cut to its first n_tokens (CPU budget)."""
    text = gdpr_text()
    if n_tokens is not None:
        text = engine.tok.decode(engine.tok.encode(text, add_special_tokens=False)[:n_tokens])
    return {"article": {"source": f"https://en.wikipedia.org/?oldid={GDPR_REVISION}", "text": text}}


# Jev's published answers for documented inputs (docs.typesafe.ai, jev-1.13.0).
HUMAN = "Is the customer asking for a human agent?"


SEVERITY = ["Cosmetic; no impact to functionality", "Broken or degraded feature, but workaround exists",
            "Blocking issue; no workaround exists"]


PY_LEVELS = ["No experience", "Some familiarity", "Regular use in a job", "Deep expertise"]


SHOES = "Shoes arrived two weeks late and in the wrong size. Also I see two charges of $120 on my card. What are you going to do about this?"


JEV_DOC_CASES = [
    # (label, state, question, Jev's tracked answer)
    ("noul: human agent", "Thanks, that fixed it!", {"type": "noul", "instructions": HUMAN}, 0.02),
    ("noul: human agent", "How do I reset my password?", {"type": "noul", "instructions": HUMAN}, 0.07),
    ("noul: human agent", "I need this sorted today, whatever it takes.", {"type": "noul", "instructions": HUMAN}, 0.26),
    ("noul: human agent", "Are you a bot?", {"type": "noul", "instructions": HUMAN}, 0.40),
    ("noul: human agent", "Is there any way to speak to someone about my invoice?", {"type": "noul", "instructions": HUMAN}, 0.84),
    ("noul: human agent", "I have asked three times now. Can I please just talk to a real person?", {"type": "noul", "instructions": HUMAN}, 0.99),
    ("noul: repeat contact", "I have asked three times now. Can I please just talk to a real person?",
     {"type": "noul", "instructions": "Has the customer contacted support about this before?",
      "criteria": {"true": "Mentions a prior attempt, ticket, or that they have asked before", "false": "No sign of any previous contact"}}, 0.93),
    ("noul: urgency", "Help! My payouts have been failing for 3 days.", {"type": "noul", "instructions": "Does this convey urgency?"}, 0.95),
    ("noul: strong python", "My experience is in Java and Go. I have not used Python.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.03),
    ("noul: strong python", "I have used Python occasionally for small scripts alongside my main Java work.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.14),
    ("noul: strong python", "I used Python every day for two years in my last job, mostly data pipelines.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.81),
    ("noul: strong python", "I have written Python daily for eight years, including maintaining a large Django codebase.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.92),
    ("noul: refund (jaggedness)", "I'm not happy with the fit. What are my options here?", {"type": "noul", "instructions": "Is the customer asking for a refund?"}, 0.22),
    ("noul: refund", "I was charged twice for the same order. Can someone look into this?", {"type": "noul", "instructions": "Is the customer asking for a refund?"}, 0.72),
    ("noul: not refund", "I was charged twice for the same order. Can someone look into this?", {"type": "noul", "instructions": "Is the customer asking for something other than a refund?"}, 0.47),
    ("score: severity", "The export button is misaligned by a few pixels on the settings page.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 0.0),
    ("score: severity", "The PDF export button does nothing when clicked. I can still export to CSV and convert it myself, but that takes ages.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.0),
    ("score: severity", "Export to PDF fails with a spinner that never finishes. Some of our team say CSV export still works for them, others say it fails too.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.11),
    ("score: severity", "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.43),
    ("score: severity", "Nobody on our team can log in since this morning. We get a 500 error on every attempt.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 2.0),
    ("score: severity, numbers-only levels", "The export button is misaligned by a few pixels on the settings page.", {"type": "score", "instructions": "Rate severity from 0 to 2, where 2 is worst", "criteria": ["0", "1", "2"]}, 0.55),
    ("score: python experience", "My experience is in Java and Go. I have not used Python.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 0.0),
    ("score: python experience", "I have used Python occasionally for small scripts alongside my main Java work.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 1.0),
    ("score: python experience", "I used Python every day for two years in my last job, mostly data pipelines.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 2.05),
    ("score: python experience", "I have written Python daily for eight years, including maintaining a large Django codebase.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 2.89),
    ("score: frustration", "Help! My payouts have been failing for 3 days.", {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}, 1.05),
    ("choice: department", "My running shoes arrived in the wrong size. Can I swap them for a size 10?", {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "returns": "Exchanges, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "returns"),
    ("choice: department", "Help! My payouts have been failing for 3 days.", {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}}, "billing"),
    ("choice: department", SHOES, {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "returns": "Exchanges, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "returns"),
    ("choice: return reason", SHOES, {"type": "choice", "instructions": "If the customer wants to return something, why?", "criteria": {
        "wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered", "damaged": "The item arrived broken or faulty",
        "changed_mind": "The item is fine, the customer no longer wants it", "other": "A return reason that fits none of the above"}}, "wrong_size"),
    ("choice: shipping issue", SHOES, {"type": "choice", "instructions": "If this is a shipping problem, which kind is it?", "criteria": {
        "not_delivered": "The package never arrived", "delayed": "The package is late but still on its way", "wrong_address": "The package went to the wrong place",
        "damaged_in_transit": "The package arrived damaged", "other": "A shipping problem that fits none of the above"}}, "delayed"),
    ("choice: resolution", SHOES, {"type": "choice", "instructions": "What does the customer want to happen?", "criteria": {
        "exchange": "Swap the item for a different one", "refund": "Money back", "replacement": "The same item sent again", "information": "Just an answer, no action needed"}}, "refund"),
    ("choice: tone", SHOES, {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}}, "frustrated"),
    ("choice: yes/no refund (jaggedness)", "I'm not happy with the fit. What are my options here?", {"type": "choice", "instructions": "Is the customer asking for a refund?", "criteria": {"yes": None, "no": None}}, "no"),
]


AG_OPTIONS = {  # AG News topics as a Choice, in label order 0..3
    "World": "World news, politics and international affairs",
    "Sports": "Sports",
    "Business": "Business, companies and the economy",
    "Technology": "Science and technology",
}


AG_QUESTION = "What is the topic of this news article?"


SUPPORT_TICKET = {
    "state": "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
            "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages", "sales": "Pricing, new plans"}},
        "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
    },
}
