# Relay

A web page that runs a work loop between an OpenAI model and Claude.

The director plans and reviews. The engineer writes code, runs it in a sandbox, and reports the real output. They trade turns until the director says DONE, the turn limit is reached, or you press Stop. You watch every turn and can stop at any point.

## What you need

- An OpenAI API key.
- An Anthropic API key.
- A free Vercel account.

Everything below happens in a browser. You do not install anything.

## Step 1: Get an OpenAI API key

1. Go to `platform.openai.com` and sign in with the same account you use for ChatGPT.
2. Open **Settings**, then **Billing**, and add a payment method. API use is billed separately from a ChatGPT subscription.
3. Open **Billing**, then **Limits**, and set a monthly budget. Start at 20 dollars.
4. Open **API keys**, then **Create new secret key**. Copy it into a note. The key is shown one time only.

## Step 2: Get an Anthropic API key

1. Go to `console.anthropic.com` and sign in.
2. Open **Billing** and buy credits. Start at 20 dollars.
3. Open **API keys**, then **Create Key**. Copy it into a note.

Both keys are separate from your ChatGPT and Claude subscriptions. Paying for a subscription does not give you API credit.

## Step 3: Deploy

1. Go to `vercel.com` and sign up with GitHub.
2. Click **Add New**, then **Project**, and import this repository.
3. Set **Root Directory** to `relay`. This step is required. The repository root holds Python files that belong to a different project, and Vercel would install those instead.
4. Expand **Environment Variables** and add both keys:
   - Name `OPENAI_API_KEY`, value your OpenAI key.
   - Name `ANTHROPIC_API_KEY`, value your Anthropic key.
5. Click **Deploy**, wait about a minute, then click **Visit**.

Vercel does not apply a changed environment variable until you redeploy. If you add a key later, open **Deployments** and click **Redeploy**.

## Step 4: Use it

1. In **What do you want done**, describe the goal in a sentence or two.
2. In **Director instructions**, paste the prompt you normally use for your director. Leave it blank for a plain director. The box remembers what you typed.
3. Press **Start**.

Each turn appears as it finishes. **Stop** ends the run after the current turn. **Continue** picks up where you left off, including after a turn limit or a network error. **Copy transcript** puts the whole exchange on your clipboard so you can paste it into ChatGPT or Claude.

## Settings

| Setting | What it does |
|---|---|
| Director model | The OpenAI model that plans and reviews. The list comes from your own key. |
| Engineer model | The Claude model that writes and runs code. Defaults to Sonnet. |
| Engineer effort | How hard the engineer thinks per turn. Low is fastest and cheapest. High is slowest. |
| Turn limit | How many turns one press of Start runs. Press Continue for more. |
| Let the engineer run code | Off means the engineer writes code but does not run it. |

## What it costs

Cost depends on the models you pick and how long the transcript grows. Published rates per million tokens:

| Model | Input | Output |
|---|---|---|
| GPT-6 Astra | $10 | $50 |
| GPT-5.6 Terra | $2 | $12 |
| Claude Sonnet 5 | $2 | $10 |
| Claude Opus 5 | $5 | $25 |

An eight-turn run with GPT-6 Astra directing and Sonnet engineering costs roughly 40 cents to a dollar. Cost per turn rises as the transcript grows, because both sides read the whole history each turn. Switching the director to GPT-5.6 Terra cuts the largest part of the bill. Start a fresh run instead of continuing a very long one.

## Limits

- The engineer runs code in a temporary sandbox. It cannot read or write your GitHub repository, and the sandbox is empty at the start of every turn. It suits self-contained experiments. For work inside a repository, use Claude Code.
- One turn must finish within 300 seconds, which is the Vercel limit on the free plan. If a turn times out, lower the engineer effort setting.
- The transcript lives in your browser tab. Closing the tab loses it. Use **Copy transcript** to keep a record.
- There is no login. Anyone with the URL can spend your API credit. Keep the URL private, or turn on Vercel Deployment Protection in **Settings**, then **Deployment Protection**.

## Running it on your own machine

```
pip install -r requirements.txt uvicorn
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
uvicorn app:app --reload
```
