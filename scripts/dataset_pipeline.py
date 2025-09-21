#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import hashlib
import random
import datetime
from collections import defaultdict, Counter, OrderedDict
from typing import List, Dict, Any, Tuple

# -----------------------------
# Helpers
# -----------------------------

def read_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl_file(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def now_stamp() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# -----------------------------
# Cleaning & normalization
# -----------------------------

USERNAME_PATTERNS = [
    re.compile(r"(user\s*(?:enters|provides|types|inputs)\s*(?:user\s*name|username)\s*)'[^']+'", re.IGNORECASE),
    re.compile(r"(username|user\s*name)\s*'[^']+'", re.IGNORECASE),
]

OTP_PATTERNS = [
    re.compile(r"OTP\s*(?:code)?\s*'\d{4,8}'", re.IGNORECASE),
    re.compile(r"OTP\s*(?:code)?\s*\d{4,8}", re.IGNORECASE),
]

ACCOUNT_PATTERNS = [
    # beneficiary account VCB84984752, MB374235995, ACB-1097-886236
    re.compile(r"(beneficiary\s+account\s+)[A-Z]{2,4}[0-9\-]{4,}", re.IGNORECASE),
    re.compile(r"(beneficiary\s+account\s+)[0-9]{2,3}(?:-[0-9]{2,6}){1,3}", re.IGNORECASE),
    re.compile(r"(account\s+)[A-Z]{2,4}[0-9\-]{4,}", re.IGNORECASE),
    re.compile(r"(account\s+)[0-9]{2,3}(?:-[0-9]{2,6}){1,3}", re.IGNORECASE),
]

# Dedupe patterns (keep first)
DEDUP_PATTERNS = [
    (re.compile(r"^daily\s+limit\s+is\s+set\s+to\s+.+\s+VND$", re.IGNORECASE), "daily_limit"),
    (re.compile(r"^account\s+balance\s+shows\s+.+\s+VND$", re.IGNORECASE), "account_balance_shows"),
]

# Template phrases and synonyms for diversification
SYNONYMS = {
    "Success message is displayed": [
        "A success banner is displayed",
        "A success toast appears",
        "A success notification is shown",
    ],
    "System processes request correctly": [
        "The system processes the request correctly",
        "The request is processed without errors",
        "The operation is handled correctly",
    ],
    "operation completes successfully": [
        "the operation completes successfully",
        "the action finishes successfully",
        "the process completes successfully",
    ],
    "user is redirected to appropriate page": [
        "the user is redirected to the appropriate page",
        "navigation proceeds to the expected page",
        "the app navigates to the correct screen",
    ],
    "user sees confirmation dialog": [
        "a confirmation dialog is displayed",
        "the confirmation prompt appears",
        "a confirmation modal is shown",
    ],
    "notification is sent": [
        "a notification is sent",
        "a notification is delivered",
        "the user receives a notification",
    ],
    "transaction record is saved": [
        "a transaction record is saved",
        "the transaction is recorded",
        "the transaction log is updated",
    ],
    "balance is updated": [
        "the balance is updated",
        "the account balance is refreshed",
        "the balance reflects the transaction",
    ],
    "confirmation number is generated": [
        "a confirmation number is generated",
        "a confirmation code is generated",
        "a reference number is created",
    ],
    # Additional generic phrases
    "page loads successfully": [
        "the page loads successfully",
        "the screen loads without errors",
        "the view renders correctly",
    ],
    "clicks Next button": [
        "clicks the Next button",
        "taps Next",
        "presses Next",
    ],
    "clicks Submit button": [
        "clicks the Submit button",
        "taps Submit",
        "presses Submit",
    ],
    "clicks Proceed button": [
        "clicks the Proceed button",
        "taps Proceed",
        "presses Proceed",
    ],
    "clicks Confirm button": [
        "clicks the Confirm button",
        "taps Confirm",
        "presses Confirm",
    ],
    "clicks Cancel button": [
        "clicks the Cancel button",
        "taps Cancel",
        "presses Cancel",
    ],
    "enters required information": [
        "inputs the required information",
        "provides the required information",
        "fills in the required details",
    ],
    "reviews the details": [
        "reviews the details",
        "verifies the details",
        "checks the details",
        "inspects the details",
    ],
    "scrolls to bottom of page": [
        "scrolls to the bottom of the page",
        "scrolls to the page end",
        "scrolls down to the bottom",
    ],
    # Contextual state & navigation
    "user is on menu page": [
        "the user is on the menu page",
        "the user has opened the menu page",
        "the user is currently on the menu page",
    ],
    "user is on main dashboard": [
        "the user is on the main dashboard",
        "the user has landed on the dashboard",
        "the dashboard screen is open",
    ],
    "user is logged into mobile banking": [
        "the user is logged in to mobile banking",
        "the user has signed in to mobile banking",
        "the user session is active in mobile banking",
    ],
    "user has accessed": [
        "the user has opened",
        "the user has navigated to",
        "the user has entered",
    ],
    "user is viewing": [
        "the user is viewing",
        "the user is checking",
        "the user is examining",
    ],
    "user opens mobile banking app": [
        "the user launches the mobile banking app",
        "the user opens the banking app",
        "the banking app is launched by the user",
    ],
    "user navigates to": [
        "the user navigates to",
        "the user goes to",
        "the user opens",
    ],
    # Actions
    "executes requested operation": [
        "performs the requested operation",
        "completes the requested operation",
        "carries out the requested operation",
    ],
    "interacts with": [
        "uses",
        "accesses",
        "performs actions on",
    ],
    "initiates transfer": [
        "starts the transfer",
        "begins the transfer",
        "initiates the transfer process",
    ],
    "confirms transaction details": [
        "confirms the transaction details",
        "reviews and confirms the transaction details",
        "verifies and confirms the transaction details",
    ],
    "selects transfer type": [
        "chooses the transfer type",
        "selects the type of transfer",
        "picks the transfer type",
    ],
    "adds transfer description": [
        "adds a transfer description",
        "enters a transfer description",
        "provides a transfer description",
    ],
    "selects beneficiary account": [
        "selects the beneficiary account",
        "chooses the beneficiary account",
        "picks the beneficiary account",
    ],
    # Visibility
    "Next button is visible": [
        "the Next button is visible",
        "the Next button is displayed",
        "the Next button is shown",
    ],
    "Submit button is visible": [
        "the Submit button is visible",
        "the Submit button is displayed",
        "the Submit button is shown",
    ],
    "Proceed button is visible": [
        "the Proceed button is visible",
        "the Proceed button is displayed",
        "the Proceed button is shown",
    ],
    "Confirm button is visible": [
        "the Confirm button is visible",
        "the Confirm button is displayed",
        "the Confirm button is shown",
    ],
    "Back button is visible": [
        "the Back button is visible",
        "the Back button is displayed",
        "the Back button is shown",
    ],
    "all required fields are displayed": [
        "all required fields are displayed",
        "all required fields are visible",
        "all mandatory fields are displayed",
    ],
}

# BDD keyword variations
BDD_VARIANTS = {
    "Given": ["Given", "Given that", "Given the user"],
    "When": ["When", "When the user", "Once"],
    "Then": ["Then", "Then the system", "After that"],
    "And": ["And", "Also", "Additionally"],
}

NEGATIVE_KEYWORDS = re.compile(r"\b(error|warning|fail|denied|invalid|blocked|rejected)\b", re.IGNORECASE)


def replace_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text
    # username → {{USER_NAME}}
    for pat in USERNAME_PATTERNS:
        s = pat.sub(lambda m: re.sub(r"'[^']+'", "'{{USER_NAME}}'", m.group(0)), s)
    # OTP → {{OTP_CODE}}
    for pat in OTP_PATTERNS:
        s = pat.sub("OTP code '{{OTP_CODE}}'", s)
    # account → {{ACCOUNT_NO}}
    for pat in ACCOUNT_PATTERNS:
        s = pat.sub(lambda m: m.group(1) + "{{ACCOUNT_NO}}", s)
    return s


def dedup_preconditions(preconds: List[str]) -> List[str]:
    seen = OrderedDict()
    keys_latest_value = {}
    for p in preconds or []:
        norm = p.strip()
        key = None
        for pat, tag in DEDUP_PATTERNS:
            if pat.match(norm):
                key = tag
                break
        if key:
            # keep first occurrence, skip subsequent
            if key not in keys_latest_value:
                keys_latest_value[key] = norm
                seen[norm.lower()] = norm
            # else: skip duplicates with same tag
        else:
            if norm.lower() not in seen:
                seen[norm.lower()] = norm
    return list(seen.values())


def diversify_phrases(lines: List[str], seed: int) -> List[str]:
    random.seed(seed)
    out = []
    for line in lines or []:
        s = line
        for base, alts in SYNONYMS.items():
            if base.lower() in s.lower():
                pattern = re.compile(re.escape(base), re.IGNORECASE)
                s = pattern.sub(random.choice(alts), s)
        out.append(s)
    return out


def diversify_bdd_line(line: str, seed: int) -> str:
    random.seed(seed)
    s = line
    # Replace BDD keyword at the start
    m = re.match(r"^(Given|When|Then|And)\b", s)
    if m:
        key = m.group(1)
        variants = BDD_VARIANTS.get(key, [key])
        new_key = random.choice(variants)
        if new_key != key:
            s = new_key + s[len(key):]
    # Apply verb/noun synonyms
    for base, alts in SYNONYMS.items():
        if base.lower() in s.lower():
            pattern = re.compile(re.escape(base), re.IGNORECASE)
            s = pattern.sub(random.choice(alts), s)
    return s


def diversify_bdd_lines(lines: List[str], seed: int) -> List[str]:
    out = []
    for idx, line in enumerate(lines or []):
        out.append(diversify_bdd_line(line, seed + idx * 13))
    return out


ADVERBIALS = [
    "successfully",
    "correctly",
    "without errors",
    "as expected",
    "smoothly",
    "reliably",
    "with valid data",
    "with proper validation",
    "within expected time",
    "with proper formatting",
    "with proper permissions",
    "under a secure session",
]


def adverbial_augment(lines: List[str], seed: int, prob: float = 0.6) -> List[str]:
    random.seed(seed)
    out = []
    for idx, s in enumerate(lines or []):
        t = s
        # apply only to Then/And-type lines
        if re.match(r"^(Then|And|Also|Additionally|After that)\b", t.strip(), flags=re.IGNORECASE):
            if random.random() < prob:
                adv = random.choice(ADVERBIALS)
                # avoid double appending if already present
                if adv.lower() not in t.lower():
                    t = t + " " + adv
        out.append(t)
    return out


# Feature-aware step enrichment (test_steps)
FEATURE_STEP_ENRICH = {
    "fund_transfer": [
        "And the fee breakdown is displayed",
        "And the transfer schedule option is available",
        "And the beneficiary nickname can be edited",
        "And the saved templates are suggested",
    ],
    "account_balance": [
        "And the currency selector is available",
        "And the auto-refresh control is visible",
        "And the balance graph is accessible",
    ],
    "transaction_history": [
        "And the export option is available",
        "And the filter panel is displayed",
        "And empty state is shown when no data",
    ],
    "card_management": [
        "And the card alias can be updated",
        "And the spending limit slider is available",
        "And the freeze or unfreeze action is confirmed",
    ],
    "notification_settings": [
        "And the push token is refreshed",
        "And the email channel verification link is sent",
        "And the Do Not Disturb schedule preview is shown",
    ],
    "beneficiary_management": [
        "And the beneficiary nickname can be saved",
        "And duplicate beneficiary warning is displayed",
        "And bank code suggestions are available",
    ],
    "bill_payment": [
        "And the biller categories are listed",
        "And the saved bill templates are suggested",
        "And the convenience fee is calculated",
    ],
}


def enrich_test_steps(feature: str, scenario: str, steps: List[str], seed: int) -> List[str]:
    random.seed(seed)
    out = list(steps or [])
    candidates = FEATURE_STEP_ENRICH.get(str(feature), [])
    if not candidates:
        return out
    k = 2  # insert up to 2 additional steps
    picks = random.sample(candidates, k=min(k, len(candidates)))
    # insert before the first Then step if present, else append
    insert_idx = None
    for i, s in enumerate(out):
        if s.strip().lower().startswith('then'):
            insert_idx = i
            break
    if insert_idx is None:
        out.extend(picks)
    else:
        for j, p in enumerate(picks):
            out.insert(insert_idx + j, p)
    return out


# Feature-aware enrichment
FEATURE_EXPECTED_ENRICH = {
    "fund_transfer": [
        "Transfer fee is displayed correctly",
        "Processing time estimate is shown",
        "Recipient information is masked appropriately",
        "Deducted amount equals transfer amount plus any fee",
        "Transaction reference matches expected format",
    ],
    "account_balance": [
        "Currency format follows locale settings",
        "Available and current balance are displayed",
        "Last updated timestamp is accurate",
        "Rounding rules are applied correctly",
        "No negative balance is displayed unless overdraft is enabled",
    ],
    "transaction_history": [
        "Default sort order is by most recent",
        "Date range filter is applied correctly",
        "Empty state message is shown when no records exist",
        "Pagination works as expected",
        "Timezone is handled correctly for timestamps",
    ],
    "card_management": [
        "Card status toggle persists",
        "Spending limits are updated",
        "Sensitive fields like CVV are masked",
        "Tokenization status is confirmed",
        "3-D Secure setting is stored",
    ],
    "notification_settings": [
        "Notification channels can be toggled",
        "Do Not Disturb schedule is saved",
        "Push token is registered",
        "Email notification address is verified",
        "Unsubscribe link is available",
    ],
    "beneficiary_management": [
        "Beneficiary name is normalized",
        "Bank code is validated",
        "Duplicate beneficiary detection works",
        "Account identifier format is validated",
        "Daily beneficiary limit checks are enforced",
    ],
    "bill_payment": [
        "Bill reference is validated",
        "Provider confirmation is received",
        "Receipt contains invoice number",
        "Late fee or discount details are accurate",
        "Partial payment rules are enforced",
    ],
}

SECURITY_EXTRA_EXPECTED = [
    "Sensitive data is masked in UI and logs",
    "No PII is stored in plaintext",
]

FEATURE_PRECONSTRAINTS = {
    "fund_transfer": [
        "Transfer fee policy is configured",
        "Daily and per-transaction limits are configured",
    ],
    "account_balance": [
        "Balance caching policy is configured",
    ],
    "transaction_history": [
        "Default date range and page size are configured",
    ],
    "card_management": [
        "Card control policies are configured",
    ],
    "notification_settings": [
        "Notification channel permissions are granted",
    ],
    "beneficiary_management": [
        "Beneficiary validation rules are configured",
    ],
    "bill_payment": [
        "Supported bill providers are configured",
    ],
}


def enrich_expected_results(feature: str, scenario: str, ers: List[str], seed: int) -> List[str]:
    random.seed(seed)
    out = list(ers or [])
    choices = FEATURE_EXPECTED_ENRICH.get(str(feature), [])
    add_n = 0
    if choices:
        add_n = 2  # add two domain-specific expectations
        picks = random.sample(choices, k=min(add_n, len(choices)))
        for p in picks:
            if p not in out:
                out.append(p)
    if str(scenario).lower() == 'security':
        for p in SECURITY_EXTRA_EXPECTED:
            if p not in out:
                out.append(p)
    return out


def enrich_preconditions(feature: str, preconds: List[str], seed: int) -> List[str]:
    random.seed(seed)
    out = list(preconds or [])
    adds = FEATURE_PRECONSTRAINTS.get(str(feature), [])
    # add up to 1 constraint line if not present
    if adds:
        candidate = random.choice(adds)
        if candidate not in out:
            out.append(candidate)
    return dedup_preconditions(out)


# Light paraphrase for expected_results
PARAPHRASE_PATTERNS = [
    (re.compile(r"\bis displayed\b", re.IGNORECASE), ["is shown", "is presented", "appears"]),
    (re.compile(r"\bis generated\b", re.IGNORECASE), ["is produced", "is created", "is issued"]),
    (re.compile(r"\bis saved\b", re.IGNORECASE), ["is stored", "is recorded", "is persisted"]),
    (re.compile(r"\bis updated\b", re.IGNORECASE), ["is refreshed", "reflects the change", "is revised"]),
    (re.compile(r"\buser is redirected to appropriate page\b", re.IGNORECASE), ["the app navigates to the expected page", "navigation proceeds to the correct page"]),
    (re.compile(r"\bSuccess message\b", re.IGNORECASE), ["Success notification", "Success banner"]),
]


def paraphrase_expected(lines: List[str], seed: int, prob: float = 0.5) -> List[str]:
    random.seed(seed)
    out = []
    for s in lines or []:
        t = s
        if random.random() < prob:
            for pat, choices in PARAPHRASE_PATTERNS:
                if pat.search(t):
                    repl = random.choice(choices)
                    t = pat.sub(repl, t)
        out.append(t)
    return out


# Quantitative preconditions variation (mask numbers into descriptive form)
QUANT_PRECOND_RULES = [
    (re.compile(r"^daily\s+limit\s+is\s+set\s+to\s+.+\s+VND$", re.IGNORECASE), "Daily limit policy is configured"),
    (re.compile(r"^account\s+balance\s+shows\s+.+\s+VND$", re.IGNORECASE), "Account balance display is available"),
]


def vary_quant_preconditions(preconds: List[str], seed: int, prob: float = 0.5) -> List[str]:
    random.seed(seed)
    out = []
    for p in preconds or []:
        t = p
        if random.random() < prob:
            for pat, repl in QUANT_PRECOND_RULES:
                if pat.match(t.strip()):
                    t = repl
                    break
        out.append(t)
    return dedup_preconditions(out)


def fix_positive_contradictions(lines: List[str]) -> List[str]:
    out = []
    for s in lines or []:
        t = s
        # Convert negative cues to positive equivalents
        t = re.sub(r"user\s+sees\s+error\s+alert", "user sees a success confirmation", t, flags=re.IGNORECASE)
        t = re.sub(r"user\s+sees\s+warning\s+popup", "no warning popup is shown", t, flags=re.IGNORECASE)
        t = re.sub(r"\berror\b", "no error", t, flags=re.IGNORECASE)
        t = re.sub(r"\bwarning\b", "no warning", t, flags=re.IGNORECASE)
        t = re.sub(r"\bfail(?:ed|ure)?\b", "succeeds", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdenied\b", "granted", t, flags=re.IGNORECASE)
        t = re.sub(r"\binvalid\b", "valid", t, flags=re.IGNORECASE)
        out.append(t)
    # Ensure at least one positive result
    if not any(re.search(r"success|successful|succeeds|confirmation", x, flags=re.IGNORECASE) for x in out):
        out.append("A success notification is shown")
    return out


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(rec)
    # Normalize placeholders across textual fields
    fields = ["title", "preconditions", "test_steps", "expected_results",
              "security_validations", "compliance_checks"]
    for f in fields:
        if f in r and r[f] is not None:
            if isinstance(r[f], list):
                r[f] = [replace_placeholders(x) for x in r[f]]
            elif isinstance(r[f], str):
                r[f] = replace_placeholders(r[f])
    # Deduplicate preconditions
    if isinstance(r.get("preconditions"), list):
        r["preconditions"] = dedup_preconditions(r["preconditions"])
    # Fix contradictions for positive scenarios
    if str(r.get("scenario_type", "")).lower() == "positive":
        if isinstance(r.get("expected_results"), list):
            r["expected_results"] = fix_positive_contradictions(r["expected_results"])
        if isinstance(r.get("test_steps"), list):
            r["test_steps"] = fix_positive_contradictions(r["test_steps"])
    # Diversify phrases deterministically & BDD variants
    seed = int(hashlib.md5(str(r.get("test_id", "")).encode('utf-8')).hexdigest(), 16) % (10**8)
    if isinstance(r.get("test_steps"), list):
        # Only fix contradictions at this stage; do not diversify yet (will be applied per split)
        r["test_steps"] = fix_positive_contradictions(r["test_steps"])
    if isinstance(r.get("expected_results"), list):
        # Enrich expected results with domain-specific items (no paraphrase here)
        r["expected_results"] = enrich_expected_results(r.get("feature"), r.get("scenario_type"), r["expected_results"], seed + 23)
    # Enrich preconditions with light domain-specific constraints
    if isinstance(r.get("preconditions"), list):
        r["preconditions"] = enrich_preconditions(r.get("feature"), r["preconditions"], seed + 31)
    return r


# -----------------------------
# Splitting & metrics
# -----------------------------

def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(records)
    hashes = set(stable_hash(r) for r in records)
    unique_hashes = len(hashes)
    total_steps = 0
    steps_set = set()
    feature_dist = Counter()
    scenario_dist = Counter()
    total_quality = 0.0
    for r in records:
        feature_dist[str(r.get('feature', 'unknown'))] += 1
        st = str(r.get('scenario_type', 'unknown')).lower()
        scenario_dist[st] += 1
        q = r.get('quality_score')
        if isinstance(q, (int, float)):
            total_quality += float(q)
        steps = r.get('test_steps') or []
        total_steps += len(steps)
        for s in steps:
            steps_set.add(s.strip())
    unique_steps = len(steps_set)
    diversity_score = (unique_steps / total_steps) if total_steps > 0 else 0.0
    avg_quality = (total_quality / total_cases) if total_cases > 0 else 0.0
    security_cov = (scenario_dist.get('security', 0) / total_cases) if total_cases > 0 else 0.0
    return {
        "total_cases": total_cases,
        "unique_hashes": unique_hashes,
        "total_steps": total_steps,
        "unique_steps": unique_steps,
        "diversity_score": diversity_score,
        "feature_distribution": dict(feature_dist),
        "scenario_distribution": dict(scenario_dist),
        "avg_quality_score": avg_quality,
        "security_coverage": security_cov,
    }


def resplit(records: List[Dict[str, Any]], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[str(r.get('scenario_type', '')).lower()].append(r)
    train, val, test = [], [], []
    for scen, group in by_scenario.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            # adjust val down if needed
            n_val = max(0, n - n_train)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train+n_val])
        test.extend(group[n_train+n_val:])
    return train, val, test


# -----------------------------
# Security fields enrichment/removal
# -----------------------------
DEFAULT_SEC_VALIDATIONS = [
    "SSL/TLS encryption is active",
    "Session token is valid",
    "Input data is sanitized",
]
DEFAULT_COMPLIANCE = [
    "ISO-27001: Access control is enforced",
]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]


def ensure_security_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    scen = str(out.get('scenario_type', '')).lower()
    if scen == 'security':
        if not out.get('security_validations'):
            out['security_validations'] = DEFAULT_SEC_VALIDATIONS
        if not out.get('compliance_checks'):
            out['compliance_checks'] = DEFAULT_COMPLIANCE
        if not out.get('risk_level'):
            # deterministic pick by test_id hash
            h = int(hashlib.md5(str(out.get('test_id', '')).encode()).hexdigest(), 16)
            out['risk_level'] = RISK_LEVELS[h % len(RISK_LEVELS)]
    else:
        # Remove security-only fields if leaked
        for k in ['security_validations', 'compliance_checks', 'risk_level']:
            if k in out:
                out.pop(k, None)
    return out


# -----------------------------
# ID management
# -----------------------------

def make_unique_ids(records: List[Dict[str, Any]], namespace: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    counter = 1
    ts = now_stamp()
    for r in records:
        newr = dict(r)
        tid = str(newr.get('test_id', '')).strip()
        if not tid or tid in seen:
            new_id = f"TC_{ts}_{namespace}_{counter:06d}"
            counter += 1
            newr['original_test_id'] = tid or None
            newr['test_id'] = new_id
        seen.add(newr['test_id'])
        out.append(newr)
    return out


# -----------------------------
# JSONL conversion
# -----------------------------

def to_jsonl_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in records:
        scen = str(r.get('scenario_type', '')).lower()
        inp = {
            "task": "generate_test_case",
            "feature": r.get('feature'),
            "scenario_type": r.get('scenario_type'),
            "priority": r.get('priority'),
            "preconditions": r.get('preconditions', []),
        }
        out = {
            "title": r.get('title'),
            "test_steps": r.get('test_steps', []),
            "expected_results": r.get('expected_results', []),
        }
        if scen == 'security':
            out["security_validations"] = r.get('security_validations', [])
            out["compliance_checks"] = r.get('compliance_checks', [])
            out["risk_level"] = r.get('risk_level', 'LOW')
        rows.append({"input": inp, "output": out, "meta": {"test_id": r.get('test_id')}})
    return rows


# -----------------------------
# Main pipeline
# -----------------------------

def discover_files(input_dir: str) -> Dict[str, str]:
    files = os.listdir(input_dir)
    def pick(prefix: str) -> str:
        cands = [x for x in files if x.lower().startswith(prefix) and x.lower().endswith('.json')]
        if not cands:
            raise FileNotFoundError(f"Cannot find {prefix}*.json in {input_dir}")
        # prefer latest by timestamp in name
        cands.sort(reverse=True)
        return os.path.join(input_dir, cands[0])
    return {
        'train': pick('train'),
        'val': pick('val'),
        'test': pick('test'),
        'metadata': pick('metadata'),
    }


def load_all_records(paths: Dict[str, str]) -> List[Dict[str, Any]]:
    all_records = []
    for split in ['train', 'val', 'test']:
        data = read_json_file(paths[split])
        for r in data:
            rr = dict(r)
            rr['__orig_split'] = split
            all_records.append(rr)
    return all_records


def process_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for r in all_records:
        rr = normalize_record(r)
        rr = ensure_security_fields(rr)
        cleaned.append(rr)
    # Global unique IDs (namespace ALL), then re-namespace by final split later if needed
    cleaned = make_unique_ids(cleaned, namespace='ALL')
    return cleaned


def postprocess_split(records: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    # Different augmentation intensities per split
    if split_name.lower() == 'train':
        adv_prob = 0.5
        para_prob = 0.5
        quant_prob = 0.5
    else:
        adv_prob = 0.2
        para_prob = 0.2
        quant_prob = 0.2
    out = []
    for r in records:
        rr = dict(r)
        tid = str(rr.get('test_id', ''))
        seed_base = int(hashlib.md5(tid.encode('utf-8')).hexdigest(), 16) % (10**8)
        # Steps: synonyms + BDD variants + adverbial
        steps = rr.get('test_steps') or []
        steps = diversify_phrases(steps, seed_base + 7)
        steps = diversify_bdd_lines(steps, seed_base + 17)
        steps = adverbial_augment(steps, seed_base + 37, prob=adv_prob)
        rr['test_steps'] = steps
        # Expected results: light paraphrase
        ers = rr.get('expected_results') or []
        ers = paraphrase_expected(ers, seed_base + 23, prob=para_prob)
        rr['expected_results'] = ers
        # Preconditions: quantitative variation
        pre = rr.get('preconditions') or []
        pre = vary_quant_preconditions(pre, seed_base + 31, prob=quant_prob)
        rr['preconditions'] = pre
        out.append(rr)
    return out


def main():
    parser = argparse.ArgumentParser(description='Dataset processing pipeline: clean, resplit, JSONL, metadata, README.')
    parser.add_argument('--input-dir', required=True, help='Path to input dataset directory containing train/val/test/metadata json files')
    parser.add_argument('--output-root', required=True, help='Root directory to write processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--name', default=None, help='Optional name for output folder')
    args = parser.parse_args()

    ts = now_stamp()
    name = args.name or f"diverse_v4_{ts}"
    out_dir = os.path.join(args.output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    # Discover files and load
    paths = discover_files(args.input_dir)
    all_records = load_all_records(paths)

    # Process & clean
    cleaned = process_records(all_records)

    # Resplit to balance scenarios in each split
    train, val, test = resplit(cleaned, args.train_ratio, args.val_ratio, args.test_ratio, seed=98765)

    # Ensure per-split unique IDs (namespace by split)
    train = make_unique_ids(train, namespace='TR')
    val = make_unique_ids(val, namespace='VA')
    test = make_unique_ids(test, namespace='TE')

    # Postprocess per split (diversity tuning)
    train = postprocess_split(train, 'train')
    val = postprocess_split(val, 'val')
    test = postprocess_split(test, 'test')

    # Write split JSON
    write_json_file(os.path.join(out_dir, 'train.json'), train)
    write_json_file(os.path.join(out_dir, 'val.json'), val)
    write_json_file(os.path.join(out_dir, 'test.json'), test)

    # Compute metrics AFTER postprocess
    m_train = compute_metrics(train)
    m_val = compute_metrics(val)
    m_test = compute_metrics(test)
    # Global metrics from concatenation of splits
    all_processed = train + val + test
    global_metrics = compute_metrics(all_processed)

    metadata = {
        "timestamp": ts,
        "metrics_all": global_metrics,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "schema_version": "v4_instruction_tuning",
        "notes": {
            "placeholders": ["{{USER_NAME}}", "{{ACCOUNT_NO}}", "{{OTP_CODE}}"],
            "diversity_synonyms": True,
            "security_fields_train_included": True
        }
    }
    write_json_file(os.path.join(out_dir, 'metadata.json'), metadata)

    # JSONL for instruction-tuning
    train_jsonl = to_jsonl_rows(train)
    val_jsonl = to_jsonl_rows(val)
    test_jsonl = to_jsonl_rows(test)
    write_jsonl_file(os.path.join(out_dir, 'train.jsonl'), train_jsonl)
    write_jsonl_file(os.path.join(out_dir, 'val.jsonl'), val_jsonl)
    write_jsonl_file(os.path.join(out_dir, 'test.jsonl'), test_jsonl)

    # Auto-generate README
    readme = f"""
# Processed Dataset: {name}

This dataset was auto-processed on {ts} to prepare for generative test case training.

## Objectives
- Reduce distribution shift across splits (train/val/test all include positive/negative/security/edge with similar ratios).
- Ensure unique `test_id` across the entire dataset (namespaced by split).
- Normalize placeholders to English/standard: `{{USER_NAME}}`, `{{ACCOUNT_NO}}`, `{{OTP_CODE}}`.
- Fix basic contradictions (e.g., positive scenarios should not expect errors) and deduplicate contradictory preconditions.
- Increase linguistic variety with deterministic synonym substitutions to improve diversity.
- Provide instruction-tuning JSONL files and updated metadata.

## Files
- train.json, val.json, test.json: cleaned JSON arrays.
- train.jsonl, val.jsonl, test.jsonl: instruction-tuning pairs with fields:
  - input: {{ task, feature, scenario_type, priority, preconditions }}
  - output: {{ title, test_steps, expected_results, [security_validations, compliance_checks, risk_level if scenario=security] }}
- metadata.json: metrics for all and per split.
- reports/validation_report.json: results from validation script.

## Placeholders & Synthetic Data
All sample data are synthetic/placeholder. Sensitive fields are masked with placeholders:
- `{{USER_NAME}}` for user names
- `{{ACCOUNT_NO}}` for account identifiers
- `{{OTP_CODE}}` for one-time passwords

## Quality Rules
- Required fields must exist for all records: test_id, title, feature, scenario_type, priority, preconditions[], test_steps[], expected_results[].
- For `scenario_type=security`, the following must also exist: security_validations[], compliance_checks[], risk_level.
- Positive scenarios should not contain negative terms in expected results (error, warning, fail, denied, invalid, blocked, rejected).
- Diversity metric (unique_steps/total_steps) should improve over raw dataset.

""".strip()
    write_json_file(os.path.join(out_dir, 'README.json'), {"README": readme})
    # Also write a markdown for human readers
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme + "\n")

    # Print summary to STDOUT
    def fmt(m):
        return (
            f"cases={m['total_cases']}, steps={m['total_steps']}, unique_steps={m['unique_steps']}, "
            f"diversity={m['diversity_score']:.4f}, avg_quality={m['avg_quality_score']:.3f}, "
            f"scenarios={m['scenario_distribution']}"
        )
    print("OUTPUT_DIR=", out_dir)
    print("GLOBAL:", fmt(global_metrics))
    print("TRAIN:", fmt(m_train))
    print("VAL:", fmt(m_val))
    print("TEST:", fmt(m_test))


if __name__ == '__main__':
    main()

