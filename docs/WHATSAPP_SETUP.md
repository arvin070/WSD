# Connecting ScamShield to a WhatsApp Business number

> The WhatsApp Business **app** (Android/iOS) has no public API and cannot be
> extended. The supported route is the **WhatsApp Business Platform (Cloud
> API)**, which is what this connects to. Unofficial libraries
> (`whatsapp-web.js`, Baileys) breach the Terms of Service and get numbers
> banned — do not use them.

## What you need first

| Requirement | Where it comes from |
| --- | --- |
| Meta Business account | business.facebook.com |
| App with WhatsApp product | developers.facebook.com → My Apps → Create App |
| A phone number registered to the app | App Dashboard → WhatsApp → API Setup |
| A **public HTTPS** URL | Meta will not call `http://` or `localhost` |

For local testing, expose your machine with a tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

## 1. Fill in `.env`

```bash
cp .env.example .env
```

| Variable | Where to find it |
| --- | --- |
| `SCAMSHIELD_WA_ENABLED` | set to `true` — nothing is mounted otherwise |
| `SCAMSHIELD_WA_APP_SECRET` | App Dashboard → Settings → Basic → App Secret |
| `SCAMSHIELD_WA_VERIFY_TOKEN` | **you invent this** — any random string |
| `SCAMSHIELD_WA_ACCESS_TOKEN` | WhatsApp → API Setup (use a permanent System User token for production) |
| `SCAMSHIELD_WA_PHONE_NUMBER_ID` | WhatsApp → API Setup, under the number |

Leave `SCAMSHIELD_WA_AUTO_REPLY=false` for the first run. Analysis still
happens; nothing is sent from your number until you decide otherwise.

## 2. Check before you wire anything up

```bash
scamshield-whatsapp-check --public-url https://your-tunnel.example.com
```

It validates every setting, confirms a model is trained and the store is
writable, and prints the exact two values to paste into the dashboard. Add
`--live` to confirm the access token and number against the Graph API:

```bash
scamshield-whatsapp-check --live --public-url https://your-tunnel.example.com
```

## 3. Start the service

```bash
make webhook-dev          # or: scamshield-serve
```

## 4. Register the webhook

App Dashboard → WhatsApp → Configuration → Edit:

- **Callback URL** — `https://your-tunnel.example.com/webhook/whatsapp`
- **Verify token** — the value of `SCAMSHIELD_WA_VERIFY_TOKEN`
- Subscribe to the **`messages`** field

Meta calls `GET` once to verify. The service echoes the challenge only when
the token matches; a mismatch returns 403 and the dashboard shows the failure.

## 5. Confirm it is live

Message the business number from another phone, then:

```bash
curl -s localhost:8000/webhook/whatsapp/stats
curl -s localhost:8000/webhook/whatsapp/contacts/<wa_id>
```

`messages` should increase by one. Send a test scam line and `flagged` should
increase too.

## 6. Turn on replies when you are ready

```bash
SCAMSHIELD_WA_AUTO_REPLY=true
```

A warning is then sent **from your business number** to any contact whose
message crosses the high-confidence threshold, threaded under the offending
message. A per-contact cooldown (`SCAMSHIELD_WA_ALERT_COOLDOWN_SECONDS`,
default 1 hour) stops a burst of scam messages producing a burst of replies.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Dashboard says verification failed | `SCAMSHIELD_WA_VERIFY_TOKEN` differs from the dashboard value |
| Deliveries return 403 | `SCAMSHIELD_WA_APP_SECRET` is wrong — signatures are checked on every request |
| Deliveries return 503 | App secret not set at all; the webhook refuses to run unauthenticated |
| `/webhook/whatsapp` is 404 | `SCAMSHIELD_WA_ENABLED` is not `true` |
| Same message analysed twice | It will not be — message ids are the store's primary key, so retries are deduplicated |
| No replies sent | `wa_auto_reply` is off, risk is below threshold, or the contact is in cooldown — `/stats` shows which |

## What this does and does not see

The Cloud API delivers only messages sent **to your business number**. It
cannot read a customer's other chats, so this protects conversations with you
— not the customer's whole inbox. Media messages (images, voice notes,
documents) are recorded but not analysed; the model reads text only.
