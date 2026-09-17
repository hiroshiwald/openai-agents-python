# Relay

Two AI models work on a goal together while you do something else. A director plans the work and reviews results. An engineer writes code, runs it, and reports what actually happened. They trade turns until they finish, until your budget runs out, or until they stop to ask you something.

Nothing runs on your computer. You do not need a browser open. Each run is a GitHub issue, so you get an email on every turn and you can reply from your phone.

## Setting it up, once

You need an OpenAI API key and an Anthropic API key. Both are separate from your ChatGPT and Claude subscriptions. Paying for a subscription gives you no API credit.

**1. Get the OpenAI key.** Go to `platform.openai.com`. Under Settings, Billing, add a payment method. Under Billing, Limits, set a monthly cap. Start at 20 dollars. Under API keys, create a key and copy it.

**2. Get the Anthropic key.** Go to `console.anthropic.com`. Under Billing, buy credit. Start at 20 dollars. Under API keys, create a key and copy it.

**3. Put both keys in this repository.** Open this repository on GitHub. Click Settings, then Secrets and variables, then Actions. Click New repository secret twice:

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key |
| `ANTHROPIC_API_KEY` | your Anthropic key |

Type the names exactly as shown. GitHub hides the values from everyone, including you, once saved.

**4. Turn on Actions.** Click the Actions tab. If GitHub asks you to enable workflows, say yes.

Setup is done. You never repeat these steps.

## Starting a run

1. Click the Issues tab, then New issue, then Start a run.
2. Fill in the form. Only the goal really matters. Everything else has a sensible default.
3. Click Create.

Work begins within about 15 minutes and you get an email on every turn. Open the issue at any time to read the whole exchange.

## The form

| Field | What it does |
|---|---|
| Goal | What you want done. One or two sentences. |
| Director instructions | How you want the director to think. Paste your usual prompt here, or leave it blank. |
| Budget in US dollars | The run stops before any turn that could push it over. The most you can set is 50. |
| Check in with me every N turns | How often the run pauses to hear from you. |
| Turn limit | The most turns this run will ever take. |
| Director model | Which OpenAI model plans the work. |
| Engineer model | Which Claude model writes and runs the code. |
| Engineer effort | How hard the engineer thinks per turn. Low is cheapest and fastest. |

## Checking in

The run pauses and waits for you in three cases: after the number of turns you set, whenever the director needs a decision from you, and whenever you ask it to. While it waits, it costs nothing.

To answer, comment on the issue. Plain words are treated as guidance for the director and the run carries on. These words are treated as commands instead:

| Comment | What happens |
|---|---|
| `STOP` | The run ends and never starts again. |
| `PAUSE` | The run waits. Comment anything to start it again. |
| `BUDGET 5` | Raises the budget to five dollars and carries on. |
| `CONTINUE` | Restarts a run that stopped at its budget or turn limit. |

You can also close the issue to end a run.

## How the budget works

Before every single turn, the relay works out the most that turn could cost. It adds that to what the run has already spent. If the total would go over your budget, the turn is not sent and the run stops with a note telling you what it would have cost.

The estimate is deliberately pessimistic. It assumes more input tokens than a turn really uses and it prices the reply at the maximum length, even though most replies are shorter. A run therefore stops a little early rather than a little late.

Real spending is added up from the token counts that OpenAI and Anthropic report after each call, and it appears at the bottom of every comment, like `Turn 6 of 20 · spent $0.31 of $2.00`.

Prices live in `relay/pricing.py`. A model that is not listed there is charged at a deliberately high rate, so an out-of-date price list makes the budget stricter rather than looser.

## What it costs

Published rates per million tokens:

| Model | Input | Output |
|---|---|---|
| GPT-6 Astra | $10 | $50 |
| GPT-5.6 Sol | $4 | $20 |
| GPT-5.6 Terra | $2 | $12 |
| GPT-5.6 Luna | $0.20 | $1.20 |
| Claude Opus 5 | $5 | $25 |
| Claude Sonnet 5 | $2 | $10 |

Cost per turn climbs as a run grows, because both models read the whole history every turn. GPT-5.6 Terra directing and Claude Sonnet 5 engineering is the cheapest sensible pairing and is the default. Short runs cost more per useful result than the budget suggests, so give a run enough turns to get somewhere.

GitHub Actions is free for public repositories. Private repositories get about 2,000 free minutes a month, and this relay uses a few minutes a day.

## Pace

A scheduled check runs every 15 minutes and moves each open run forward by up to 4 turns. Commenting on an issue starts a check straight away, so a reply from you is acted on within a minute or two.

GitHub runs scheduled jobs when it has capacity, so a check can be late or skipped when GitHub is busy. A skipped check costs you nothing and the next one picks up where the last left off.

## Limits

- The engineer runs code in a fresh sandbox each turn. It cannot read or write this repository or any other, and nothing it writes survives to the next turn. It suits self-contained experiments. For work inside a codebase, use Claude Code.
- Anyone who can see this repository can read your runs and spend your API credit by opening an issue. Keep the repository private.
- GitHub switches off scheduled workflows in a repository that has had no activity for 60 days. An active relay keeps itself awake. If it does go quiet, open the Actions tab and click Enable.

## If something looks wrong

| What you see | What it means |
|---|---|
| Nothing happens after you open an issue | Actions is off. Open the Actions tab and enable workflows. |
| A comment saying a key is not set | A secret is missing or its name is misspelled. Check Settings, Secrets and variables, Actions. |
| A comment about a failed turn | The run tries again on the next check. Open the Actions tab to read the full error. |
| The run stops sooner than you expected | It hit the budget. The comment says what the next turn would have cost. Comment `BUDGET 5` to carry on. |
