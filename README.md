# ScamShield

Scam detection for WhatsApp conversations — a trained ML classifier, a rule
engine that explains *why* a message was flagged, and a web app that analyses a
whole exported chat rather than one message at a time.

Everything runs locally. No message is sent anywhere.

---

## Quick start

```bash
make setup    # virtualenv + editable install
make train    # ~25s, writes models/
make serve    # http://127.0.0.1:8000
```

Without `make`:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/scamshield-train
./.venv/bin/scamshield-serve
```

Docker:

```bash
make docker-build && make docker-run
```

To get a chat out of WhatsApp: **⋮** → **More** → **Export chat** → **Without
media**, then upload the `.txt` or paste it in.

---

## Architecture

```
src/scamshield/
├── config.py            all tunables, from env / .env (SCAMSHIELD_*)
├── logging_setup.py     plain or JSON logging
├── resources.py         typed loaders for the YAML content files
├── resources/
│   ├── lexicons.yaml    374 terms, 14 signal families, weights, scoring constants
│   └── templates.yaml   123 scam templates, 247 ham messages, slot vocabularies
├── core/                framework-free domain logic
│   ├── normalizer.py    NFKD + homoglyph folding; reports what it undid
│   ├── business.py      account standing, brand-identity claims
│   ├── links.py         offline URL risk analysis (no network)
│   ├── signals.py       regex primitives + directionality patterns
│   ├── features.py      57 heuristic features + context/direction discounts
│   └── chat_parser.py   Android + iOS exports, multi-line, system events
├── ml/
│   ├── corpus.py        slot filling, augmentation, external-CSV loader
│   ├── datasets.py      fetches third-party corpora, template-grouped holdout
│   ├── pipeline.py      TF-IDF (word + char) ⊕ heuristics → classifier
│   ├── calibrator.py    learned meta-fusion (opt-in, see Results)
│   ├── train.py         grouped CV, dev tuning, held-out reporting
│   └── detector.py      scoring, verdict bands, categories
├── analysis/
│   └── conversation.py  per-sender profiles, escalation, timeline
└── api/
    ├── app.py           application factory
    ├── routes.py        thin HTTP layer
    ├── schemas.py       request/response models
    └── static/          single-file web UI
```

Layers depend downward only: `api → analysis → ml → core → resources/config`.
Nothing in `core/` imports FastAPI or sklearn estimators, so the detection
logic is testable and reusable without a web server.

**Content is not code.** Lexicons and corpus templates live in YAML, because
they change whenever a new scam appears and should be editable by someone who
does not write Python. Adding a term or a template needs no code change —
edit the YAML and retrain.

---

## How the model works

Three components feed one score.

**1. TF-IDF text model.** Word 1–2 grams plus `char_wb` 3–5 grams. Character
n-grams survive leetspeak and typo evasion (`0TP`, `acc0unt`). URLs collapse to
a single token and long digit runs are masked, so the vectoriser cannot
memorise a phone number or order id from the training data.

**0. Unicode normalisation.** Every text path — lexicon matching, rule
scoring and TF-IDF tokenisation — starts by folding script spoofing: NFKD,
plus an explicit confusable map (Cyrillic `а`→`a`, Greek `ο`→`o`), Unicode
digits to ASCII, and zero-width characters removed. Combining marks are
stripped only after Latin bases, because Indic vowels *are* combining marks
and blanket-stripping them would destroy the Devanagari half of the corpus.

The normaliser returns the folded text **and a record of what it had to
undo**. Ordinary messages need no folding at all, so a Cyrillic letter inside
a Latin word is itself strong evidence — discarding it to get clean text would
lose signal. Evasion therefore *raises* risk rather than hiding it: on the
adversarial suite, spoofed scams score 0.988 against 0.680 for the same
message in plain text.

**2. Heuristic features** — 57 signals covering urgency, threats, prize claims,
credential requests, payment asks, investment pitches, impersonation, secrecy,
look-alike domains, link shorteners, raw-IP links, UPI handles, hidden
zero-width characters, character entropy (random URL slugs and padded
payloads sit near 1.0, prose well below), and interaction terms
(urgency × link, prize × fee, threat × credential).

**Directionality.** A scam *instructs* the reader ("share the OTP"); a worried
user *asks about* the same thing ("should I share my OTP?") in identical
words. Reader inquiries are discounted — but a second-person directive dressed
as a question ("Can you share the OTP?") is exempt, otherwise a single
question mark would be a trivial bypass. Both directions are tested.

These also carry **context features that push the score down**. A friend
saying *"someone called asking for my otp, I blocked them"* contains the same
keywords as the attack it describes. So does a genuine bank alert that says
*"never share this"*. The rule score is discounted when a message makes no
request of the reader — and the discount is cancelled when it does, because
real scams borrow the reassuring wording:

```
"Do not share this OTP with anyone... please forward it to this number"  → 0.75  (ask present)
"someone called asking for my otp, I disconnected"                       → 0.19  (no ask)
```

**3. Fusion.** `0.85 × model + 0.15 × rules`, tuned rather than guessed, under
the invariant that the rule weight stays below the decision threshold — a
single rule hit must never cross the line unaided, or the rules' standalone
precision (0.76) becomes the system's ceiling. A
guard caps messages under 4 tokens with no red flags — `"Hi"` is not evidence
of anything, and without it the model extrapolates from openers that are
common in scam templates.

A second model predicts *which* of 17 playbooks a flagged message follows.

---

## Conversation analysis

A per-message classifier misses what makes a chat a scam:

| Signal | Why it matters |
| --- | --- |
| Per-sender risk | Scores participants separately, so the *victim* is not flagged alongside the scammer |
| Escalation | Compares **peak** risk early vs late — means get diluted by chatty filler around the ask |
| Unsaved number | Sender stored as a raw number was never in the user's contacts |
| Burst timing | Rapid scripted sends |
| Account standing | Verified vs unverified business, and whether the claimed identity matches |
| Category rollup | Which playbook dominates the thread |

### Business accounts

Most WhatsApp Business accounts are exactly what they claim to be, so "is a
business" is never treated as evidence of a scam — that would flag every
merchant a user talks to. What counts is the gap between the identity an
account *claims* and the standing it actually *has*.

WhatsApp exports record that standing, and Meta distinguishes a plain
*business account* from an **official** (verified) one:

| Scenario | Result |
| --- | --- |
| Unverified account claiming to be HDFC, asking for an OTP | **100% — dangerous**, identity mismatch named |
| Official business account, ordinary delivery update | 0% — quietened, badge cited |
| Official business account asking for an OTP | **87% — flagged**; the badge does not excuse it |
| Unverified café confirming a table booking | 0.6% — not penalised for being unverified |

Brand *claims* are separated from brand *mentions*: "I am from Amazon support"
is an identity assertion, "I ordered from Flipkart yesterday" is a customer
talking about their order. Only the former counts.

**The badge cannot be forged.** Genuine account notices are emitted by
WhatsApp and carry no sender; anything a participant types has one. A scammer
who types "You're messaging with an official business account" into the chat
gets no trust from it, and the attempt is still scored as a normal message —
there are tests for both halves of that.

On the bundled sample chat the opener (`Hi`) scores 0% and the payload
(`share the OTP just sent to your phone`) scores 93%. The thread is flagged at
**100% with escalation detected**, while the victim's side scores **10%**.

---

## Scam playbooks

17 categories. Three were added because each has a *legitimate twin* sharing
most of its vocabulary, which is what makes them hard:

| Playbook | The scam | Its legitimate twin |
| --- | --- | --- |
| **Course / mentorship** | "Batch closes tonight, 3 students crossed 1 lakh, lifetime access ₹4999" | A real workshop with limited seats and a fee |
| **Extortion / ransom** | Sextortion, ransomware, kidnapping hoax — a threat paired with an untraceable payment | A genuine security notice about a new sign-in |
| **Trusted-contact link** | A friend-shaped message pushing a deceptive link — how compromised accounts spread | A friend sharing a recipe, an article, a cat video |

The third is the delicate one. **Friends share links constantly**, so vouching
(`"check this out"`, `"is this you in this video?"`) carries almost no weight
on its own — it fires only when the link is *structurally* suspect, which is
where the link analyser below does the real work:

```
"check this out https://www.youtube.com/watch"        → 0.034   (clean link, no signal)
"check this out http://cash-app-free.buzz/get"        → 0.966   (vouched_risky_link)
```

Extortion works the same way: a threat alone may be bluster and a payment
request alone may be ordinary, so `ransom_payment` needs the pairing. A course
advert only becomes `course_upsell` once a fee is attached.

Before these categories existed the model still caught most of these messages
but *mislabelled* them — extortion was reported as "Romance / long-con fraud",
and shared malicious links had no category at all.

---

## Link protection

Two layers, and the distinction between them matters.

### Offline analysis — always on, zero risk

Every link in every scored message is decomposed and judged **without any
network request**, so it runs automatically and cannot tell an attacker their
target is live. The UI shows the verdict and the real destination before the
user decides whether to tap.

What it catches, all from the text of the URL alone:

| Deception | Example |
| --- | --- |
| Credentials in the authority | `https://paypal.com@evil.xyz` — reads as PayPal, goes to evil.xyz |
| Brand as a subdomain | `paypal.com.secure-login.xyz` — owned by whoever owns `secure-login.xyz` |
| Brand inside the domain | `sbi-kyc-verify.xyz` — uses the name, is not `onlinesbi.sbi` |
| Typosquatting | `arnazon.in` — two characters from `amazon.in` |
| Numeric hosts | `2130706433`, `0x7f000001` — both are `127.0.0.1` |
| IDN homographs | `xn--80ak6aa92e.com` → `аррӏе.com` (Cyrillic) |
| Smuggled redirects | `?url=http%3A%2F%2Fevil.xyz`, including base64 |
| Installable payloads | `.apk`, `.exe`, `.scr`, `.jar` |
| Executable schemes | `javascript:`, `data:`, `vbscript:` |

Legitimate domains score **0.00** — `amazon.in`, `onlinesbi.sbi`,
`pay.google.com`, `irctc.co.in` are all clean. A checker that flags real
bank sites teaches people to ignore it, so the allowlist is load-bearing and
tested in both directions.

```bash
curl -s localhost:8000/api/links/analyze \
  -H 'Content-Type: application/json' -d '{"url":"http://paypal.com@sbi-kyc-verify.xyz/login"}'
```

### Sandboxed inspection — opt-in, real request

`POST /api/links/inspect` fetches the URL in isolation to answer what static
analysis cannot: where a shortener actually lands, and whether the page at the
end asks for a password.

**It is off by default, and that is a safety decision, not a default-config
oversight.** Fetching an attacker's URL confirms your number is live and
reveals your IP. Nothing about the design makes that free, so it never runs
automatically on inbound messages.

The guards, in `sandbox/guard.py`:

- **Scheme allowlist** — http and https only.
- **Every resolved address checked**, not just the first: private, loopback,
  link-local, CGNAT and reserved ranges are refused. `169.254.169.254` is the
  cloud metadata endpoint that leaks instance credentials.
- **Per-hop revalidation** — a public URL can redirect to `127.0.0.1`, so the
  check runs again on each hop rather than once at the start.
- **No execution.** No JavaScript, no downloads, nothing written to disk, no
  cookies or credentials sent, redirects followed manually.
- Timeout, redirect cap and byte cap on every fetch.

**This is an inspector, not a detonation chamber.** It reads headers, the
redirect chain, the page title and whether a login form is present. Analysing
genuine malware needs a disposable VM, which no library can honestly claim to
be, and the README would rather say so than imply otherwise.

**Known limitation:** the resolve-then-connect gap allows DNS rebinding — a
name that answers publicly for the check and privately for the connection.
Closing it needs connection-level IP pinning. The exposure is bounded because
nothing fetched is executed, stored, or forwarded.

---

## Live integration — WhatsApp Business Platform

> The WhatsApp Business **app** (the Android/iOS app) has no public API and
> cannot be extended. The supported path is the **WhatsApp Business Platform
> (Cloud API)**, which is what this integrates with. Unofficial libraries
> (Baileys, whatsapp-web.js) breach the Terms of Service and get numbers
> banned — they are deliberately not used.

Point your Cloud API webhook at `POST /webhook/whatsapp` and every inbound
customer message is scored in the thread it belongs to.

```bash
cp .env.example .env                                   # fill in the four WA_ values
scamshield-whatsapp-check --live --public-url https://your.domain
make webhook-dev
```

`scamshield-whatsapp-check` is the pre-flight: it validates every setting,
confirms a model is trained and the store is writable, optionally verifies the
token and number against the Graph API, and prints the exact Callback URL and
Verify token to paste into the Meta dashboard — so those values cannot drift
between `.env` and the dashboard. Secrets are masked in its output.

Full walkthrough, including tunnelling a local server and the troubleshooting
table: **[docs/WHATSAPP_SETUP.md](docs/WHATSAPP_SETUP.md)**.

| Endpoint | Purpose |
| --- | --- |
| `GET /webhook/whatsapp` | Meta's subscription handshake (`hub.challenge`) |
| `POST /webhook/whatsapp` | Message deliveries — signature-checked |
| `GET /webhook/whatsapp/stats` | Counters: messages, contacts, flagged, alerts |
| `GET /webhook/whatsapp/contacts/{wa_id}` | Risk history for one contact |

**Nothing is mounted unless `SCAMSHIELD_WA_ENABLED=true`.** A default install
exposes no public callback endpoint.

### The three things that break webhook integrations

**Forged deliveries.** The URL is public, so every POST is authenticated
against `X-Hub-Signature-256` — an HMAC-SHA256 of the *raw* body keyed by the
app secret, compared in constant time. Re-serialising the parsed JSON would
change the bytes and break the check, so the raw body is used. With no app
secret configured the endpoint refuses to run rather than accepting anything.

**Retries.** Meta redelivers until it receives a 2xx, so the same message
arrives repeatedly. Message ids are the primary key of the store, which makes
the first write the only one that proceeds — a retry is recorded as a
duplicate and never re-scored or re-alerted.

**Unwanted replies.** A reply goes out from *your* business number, so
`wa_auto_reply` is **off by default**. When enabled, a warning threads under
the offending message, and a per-contact cooldown stops a burst of scam
messages producing a burst of replies.

### Thread-aware, not message-at-a-time

A webhook delivers one message; the analyser reasons about conversations.
Recent history per contact is kept in SQLite and replayed on each delivery, so
escalation detection and sender profiling work on a live stream exactly as
they do on an exported chat — the long-con that opens with "Hi" is still
caught when the ask arrives days later.

Customer messages are personal data: phone numbers are masked in logs
(`***3210`), message bodies are never logged, tokens are `SecretStr`, and
`data/conversations.db` is git-ignored. `ConversationStore.prune()` trims
history to the retention window.

---

## Results

Measured on `data/realworld.jsonl` split `test` — 60 hand-written messages
sharing no templates with the generator, written before any tuning and never
read by the training loop.

Two independent held-out sets, neither read during training.

**A. Hand-written test** — 60 WhatsApp-style messages, written before any tuning.

| | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| **Fixed blend (shipped)** | 0.968 | **1.000** | 0.984 | 1.000 |
| Model only | 0.968 | 1.000 | 0.984 | 1.000 |
| Rules only | 0.786 | 0.733 | 0.759 | 0.780 |

**B. External holdout** — 3,855 third-party messages, split by *template
signature* so no template appears on both sides: **P 1.000, R 0.999, F1 1.000**.

Grouped 5-fold CV on the 37,495-row corpus: **F1 0.938, PR-AUC 0.981**.
Category model: **59% top-1, 74% top-2** over 17 playbooks — up despite three
extra classes, which is where the added data shows.

### Reading these numbers

- **The external holdout's 1.000 is not real-world proof.** Its training and
  test halves come from the *same generator*; splitting by template stops
  memorisation but not shared vocabulary and phrasing. It shows the model
  generalises across templates within one source — a real but narrow claim.
- The rule engine improved markedly once link analysis fed into it
  (standalone F1 0.691 → 0.759), since a deceptive URL is now scored
  structurally rather than by keyword.
- Recall is 1.000 on both sets, which is the number the F₂ objective targets:
  a missed scam costs the user money, a false alarm costs a moment's doubt.

**Adversarial suite** (`data/adversarial.jsonl`, 8 scams × 5 evasion variants):

| Attack | Detected | Mean risk |
| --- | --- | --- |
| none (baseline) | 6/8 | 0.680 |
| homoglyph | **8/8** | 0.988 |
| zero-width | **8/8** | 0.999 |
| fullwidth | 6/8 | 0.680 |
| combined | **8/8** | 1.000 |

Fullwidth folds to exactly the clean text and is scored identically — correct,
since fullwidth is ordinary in CJK contexts and is not evidence of anything.

### The learned meta-calibrator is built but not enabled

`ml/calibrator.py` stacks a logistic regression on `[ml_score, rule_score]`
plus nine context features, fitted on **out-of-fold** base predictions (fitting
it on in-sample predictions would teach it that `ml_score` is near-perfect,
because on training rows it is).

It lost to the fixed blend on dev — F2 0.980 vs 0.995 — so training selected
`fusion="blend"` and the calibrator sits unused behind that guard. Its learned
weights are still informative and sane: `n_zero_width` +3.99, `n_homoglyphs`
+5.08, `n_narrative` −2.05. A more sophisticated component that measures worse
is not an upgrade, so it ships switched off; `metadata.json` records both
scores, and it turns on automatically if it ever wins.

Scam-category model: **47% top-1, 61% top-2** over 14 classes under the same
grouped CV — well above the 7.1% chance baseline, but the playbooks genuinely
overlap (a fake refund is also bank phishing), so the UI hides the category
when the posterior is below 0.25.

### Read these numbers with care

- **The test set is 60 messages and I wrote them.** It shows the model
  generalises past its own templates. It is not a production accuracy
  estimate, and it shares my blind spots about what scams look like.
- **No corpus here is intercepted traffic.** The generator is ours; the Indian
  SMS set is someone else's generator. Only the UCI ham is genuinely
  real-world. Real labelled WhatsApp data remains the highest-value addition.
- Threshold 0.15 was chosen on the dev split by **F₂**, not F₁: a missed scam
  costs the user money, a false alarm costs a moment's doubt.
- Adding the normalisation and entropy features moved the test score from
  F1 0.983 to 0.967 — **a one-message difference on a 60-message set**, which
  is noise, not a trend. It is also the wrong number to optimise here: the
  test set contains no spoofed messages, so it can measure the *cost* of these
  features and never their *benefit*. That is what the adversarial suite above
  is for.

### Training data

```bash
scamshield-fetch-data    # downloads third-party corpora into data/external/
scamshield-train
```

| Source | Rows | What it is |
| --- | --- | --- |
| Synthetic generator | 16,961 | WhatsApp-style conversations, 17 playbooks |
| `CloveAI/india-spam-sms` (MIT) | 19,230 | Indian SMS, on-domain scams — **generated, not intercepted** |
| `ucirvine/sms_spam` (UCI) | 5,159 | Genuine human SMS; ham imported in full, spam tagged separately |

`alusci/sms-otp-spam-dataset` was **rejected**: its `label` is OTP validity and
its `status='spam'` is a delivery outcome, not a content label.

Any CSV with `text` and `label` columns dropped into `data/external/` is folded
in automatically.

**Domain shift is real and had to be managed.** Adding 20k SMS-style messages
first *cost* recall on conversational grooming — the long-con openers WhatsApp
is actually used for, which SMS corpora contain nothing like. Two corrections
fixed it: per-category weights in `templates.yaml` so conversational playbooks
outnumber transactional ones, and more grooming templates. Recall went
0.933 → 1.000.

The same shift broke OTP detection in a subtler way: SMS corpora are full of
legitimate OTP *deliveries*, which taught the model that OTP text is benign and
dropped a bare "share the OTP sent to your phone" to 0.24. The fix was to make
the deliver-versus-request contrast explicit in both the corpus and the dev set.

---

## Configuration

Every tunable is an environment variable (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCAMSHIELD_HOST` / `_PORT` | `127.0.0.1` / `8000` | Bind address |
| `SCAMSHIELD_CORS_ORIGINS` | loopback only | Comma-separated allowed origins |
| `SCAMSHIELD_DATA_DIR` / `_MODELS_DIR` | `./data` / `./models` | Where data and artifacts live |
| `SCAMSHIELD_THRESHOLD_OVERRIDE` | *(learned)* | Force a decision threshold |
| `SCAMSHIELD_LOW_INFO_TOKENS` | `4` | Below this word count, cap the score |
| `SCAMSHIELD_MAX_UPLOAD_BYTES` | `8388608` | Upload limit |
| `SCAMSHIELD_LOG_JSON` | `false` | JSON logs for shipping |

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/analyze/message` | `{"text": "..."}` → score, verdict, category, signals |
| `POST /api/analyze/messages` | `{"texts": [...]}` → batch |
| `POST /api/analyze/chat` | `{"content": "<export text>"}` → full conversation analysis |
| `POST /api/analyze/upload` | multipart `.txt` upload |
| `GET /api/model` | model card: thresholds, bands, corpus size, held-out metrics |
| `POST /api/links/analyze` | one URL, offline |
| `POST /api/links/scan` | every link in a message, offline |
| `POST /api/links/inspect` | sandboxed fetch (opt-in) |
| `GET /api/health` | liveness + whether a model is loaded |
| `POST /webhook/whatsapp` | Cloud API deliveries (see above) |

Interactive docs at `/docs`.

```bash
curl -s localhost:8000/api/analyze/message \
  -H 'Content-Type: application/json' \
  -d '{"text":"Your KYC expired, account blocked today. Verify: http://sbi-kyc.xyz"}'
```

---

## Development

```bash
make test     # 284 tests
make lint     # ruff
make fmt      # ruff --fix
```

---

## Limitations

- Text only. Images, voice notes and forwarded documents are not analysed.
- Business-account standing is read from the export's system line. A chat
  pasted as plain text carries no such line, so account type falls back to
  "unknown" and only the message-level identity claims apply.
- Homoglyph coverage is a curated confusable map, not the full Unicode
  confusables table. It covers Cyrillic, Greek and the Latin variants seen in
  real phishing; exotic scripts would need extending.
- Tuned for English and Hinglish/Devanagari, with an India-heavy scam
  vocabulary (UPI, KYC, IRCTC, digital arrest). Other regions need their own
  lexicon and templates — which is why they are YAML.
- Live analysis requires the **Cloud API**, which means a Meta Business
  account and a registered number. The consumer WhatsApp Business app cannot
  be integrated with — no API exists for it.
- The Cloud API only delivers messages sent to *your* business number. It
  cannot see a customer's other chats, so this protects conversations with
  you, not the customer's whole inbox.
- A determined attacker who knows the model can word around it. This is an
  assistive signal, not a guarantee.
