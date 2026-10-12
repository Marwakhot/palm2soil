# Orbit backend architecture

How the Orbit backend is put together. Start with the overview, then open the flow for the feature you care about. The two tables are the reference for which checks each endpoint runs and who uses which store and model. File paths are relative to `src/`, except `scripts/`.

## 1. Overview

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 400}}}%%
flowchart TB
    Client(("Mobile app"))
    API["HTTP API<br/>index.ts<br/>routes/orbit.ts · routes/maps.ts · routes/baitna.ts<br/>middleware/auth.ts · middleware/rate-limit.ts<br/>middleware/error-handler.ts<br/>lib/circuit-breaker.ts · lib/analytics.ts"]

    subgraph Features["Feature modules"]
        direction LR
        Maps["Maps<br/>services/places-service.ts<br/>services/routes-service.ts"]
        Chat["Orbit chat<br/>services/orbit-service.ts<br/>services/llm.ts<br/>lib/model-router.ts<br/>services/retrieval.ts"]
        Plan["Day planner<br/>services/plan-service.ts"]
        Index["Baitna indexer, every 5 min by default<br/>baitna/scheduler.ts<br/>baitna/indexer.ts<br/>baitna/chunk.ts"]
        Baitna["Baitna housing assistant<br/>baitna/service.ts<br/>helpers: see section 5"]
    end

    Google["Google Maps Platform<br/>Places + Routes"]

    subgraph Platform["Shared platform"]
        direction LR
        Shared["Shared code<br/>services/conversation.ts<br/>services/embedding.ts<br/>config + types, used everywhere:<br/>config.ts · types.ts<br/>baitna/config.ts · baitna/types.ts"]
        OpenRouter["OpenRouter<br/>LLMs + embeddings"]
        Redis[("Redis, in-memory fallback<br/>lib/redis.ts")]
        Supabase[("Supabase, Postgres + Auth<br/>lib/supabase.ts")]
    end

    AppBackend["Student app backend<br/>partners · listings · eligibility<br/>client: baitna/app-api.ts"]

    Client --> API
    API -->|"/maps/*"| Maps
    API -->|"/orbit/chat"| Chat
    API -->|"/orbit/plan"| Plan
    API -.->|"/orbit/baitna/reindex"| Index
    API -->|"/orbit/baitna/chat"| Baitna

    Maps --> Google
    Chat --> Google
    Plan --> Google
    Chat --> Platform
    Plan --> Platform
    Baitna --> Platform
    Index --> Platform
    Baitna --> AppBackend

    classDef app fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef data fill:#ffe4e6,stroke:#e11d48,color:#881337
    classDef ext fill:#e0e7ff,stroke:#4f46e5,color:#312e81
    class Client,API,Chat,Plan,Maps,Baitna,Index,Shared app
    class Supabase,Redis data
    class OpenRouter,Google,AppBackend ext
```

Almost everything uses the shared platform, so the picture draws one arrow to the group instead of one per service. [Who uses what](#6-who-uses-what) has the details.

## 2. Request handling

Every endpoint, and the checks it runs before the feature code.

| Endpoint | Auth | Rate limit | Circuit breaker | Handled by |
|---|---|---|---|---|
| `POST /orbit/chat` | yes | yes | yes | `services/orbit-service.ts`: SSE stream by default, JSON if the client asks |
| `POST /orbit/plan` | yes | yes | no | `services/plan-service.ts` |
| `POST /orbit/plan/:id/save` | yes | no | no | route reads the cached plan from Redis and writes `saved_plans` |
| `GET /orbit/plan/shared/:token` | **none** | no | no | route reads `saved_plans` |
| `GET /maps/places/autocomplete`, `GET /maps/places/details` | yes | no | no | `services/places-service.ts` |
| `POST /maps/routes` | yes | no | no | `services/routes-service.ts` |
| `POST /orbit/baitna/chat` | yes | yes | yes | `baitna/service.ts` |
| `POST /orbit/baitna/reindex` | shared secret (`x-reindex-secret`) | no | no | `baitna/scheduler.ts` |
| `GET /`, `GET /orbit/health`, `GET /orbit/baitna/health` | none | no | the two `/health` routes report it | route |

- **Auth** (`middleware/auth.ts`) checks the bearer token with Supabase Auth and loads the `users` row. If the request sends an `X-Device-ID` header, the first ID seen is bound to the user and a different one later gets a 403. Requests without the header skip the device check.
- **Rate limit** (`middleware/rate-limit.ts`) allows 10 requests per 60 s and 150 per day for each user by default (`VELOCITY_LIMIT`, `VELOCITY_WINDOW`, `DAILY_CHAT_LIMIT`). Orbit chat, Orbit plan and Baitna chat share both counters. The counters live in Redis.
- **Circuit breaker** (`lib/circuit-breaker.ts`) is a single in-process breaker shared by Orbit chat and Baitna chat. It opens when 5 different users have failed with no successful request in between. While it is open (60 s), both chats and both `/health` routes answer 503. Orbit chat only counts timeouts as failures, because `services/orbit-service.ts` catches its other errors and replies itself. Baitna chat counts every error except rate limits and 4xx answers from the student app backend.
- **Feature flag:** when `FEATURE_SV_ORBIT_ENABLED` is off (it is on by default), Orbit chat, Baitna chat and both `/health` routes answer 503. Plans, maps and shared links keep working. Baitna chat also answers 404 when the student app backend reports Baitna as switched off.
- **Analytics** (`lib/analytics.ts`): both chat routes write an `analytics_events` row without waiting for it. Baitna chat logs only successful replies. Orbit chat also logs requests that ended in an error reply, because the service handles those errors itself.

## 3. Orbit chat

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 400}}}%%
flowchart TD
    Start(["POST /orbit/chat<br/>routes/orbit.ts → services/orbit-service.ts"]) --> History["Load last 10 messages<br/>services/conversation.ts"]
    History --> Attached{"Images or place<br/>attached?"}
    Attached -->|yes| Conv
    Attached -->|no| Mode{"Request mode"}
    Mode -->|"chat or find"| Intent["Classify intent<br/>free model, keyword fallback<br/>services/llm.ts · lib/prompts.ts"]
    Mode -->|plan| Search
    Intent -->|"offer intent"| Search
    Intent -->|"anything else"| Conv

    Conv["Pick model by complexity<br/>free / standard / premium / vision<br/>lib/model-router.ts"] --> Reply["Generate reply<br/>services/llm.ts<br/>system prompt: lib/prompts.ts"]

    Search["Vector search for deals<br/>services/retrieval.ts → services/embedding.ts<br/>Supabase RPC"] --> Cards["Offer cards + one-line intro (free model)<br/>or a no-deals reply<br/>services/orbit-service.ts · services/llm.ts · lib/distance.ts"]
    Cards --> FindMode{"JSON request in find<br/>mode with location?"}
    FindMode -->|yes| Pins["Add nearby Google Places<br/>as map pins<br/>services/places-service.ts"]
    FindMode -->|no| Save
    Pins --> Save
    Reply --> Save["Save turn<br/>services/conversation.ts"]
    Save --> Done(["Respond: SSE stream by default (services/llm.ts),<br/>JSON if requested (routes/orbit.ts)"])

    classDef step fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef io fill:#dbeafe,stroke:#2563eb,color:#172554
    class History,Intent,Conv,Reply,Search,Cards,Pins,Save step
    class Start,Done io
```

## 4. Day planner

```mermaid
sequenceDiagram
    actor App as Mobile app
    participant API as routes/orbit.ts
    participant Plan as services/plan-service.ts
    participant Deals as services/retrieval.ts
    participant Places as services/places-service.ts
    participant Routes as services/routes-service.ts
    participant Redis as Redis (lib/redis.ts)
    participant DB as Supabase (lib/supabase.ts)

    App->>API: POST /orbit/plan (time window, budget, vibe, location)
    Note over API: middleware/auth.ts, window of at least 2 hours, middleware/rate-limit.ts
    API->>Plan: buildPlan()
    par in parallel
        Plan->>Deals: deal vector search (services/embedding.ts + Supabase RPC)
    and
        Plan->>Places: nearby places, if location given
    end
    Note over Plan: places only top up the list when there are fewer than 3 deals
    Note over Plan: must-include places are scheduled first
    loop until the day is full or nothing else fits
        Plan->>Plan: pick the nearest stop that fits the time and budget (lib/distance.ts)
        Plan->>Routes: live route for this leg (Google Routes)
        Note over Plan: straight-line estimate if routing fails (lib/distance.ts)
    end
    Note over Plan: if nothing affordable fits, an empty plan with a suggested budget
    Plan->>Redis: cache plan for 1 hour
    Plan-->>API: itinerary, budget, map pins
    API-->>App: plan

    App->>API: POST /orbit/plan/:id/save (middleware/auth.ts)
    API->>Redis: read cached plan
    API->>DB: insert into saved_plans
    API-->>App: share URL

    App->>API: GET /orbit/plan/shared/:token (no auth)
    API->>DB: read saved_plans
    API-->>App: plan
```

## 5. Baitna housing assistant

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 400}}}%%
flowchart TD
    Start(["POST /orbit/baitna/chat<br/>routes/baitna.ts → baitna/service.ts"]) --> Scope["Record scope<br/>a scope change clears history + constraints<br/>baitna/session.ts"]
    Scope --> Classify["Classify intent<br/>free model, 4 s timeout, regex fallback<br/>baitna/classify.ts · baitna/intent-rules.ts"]
    Scope --> Partners["Fetch partners + live listings<br/>student app backend<br/>baitna/app-api.ts"]
    Partners --> Guard{"Partner scope: partner<br/>inactive or another<br/>partner named?<br/>baitna/service.ts · baitna/guards.ts"}
    Guard -->|yes| Redirect(["Short redirect reply<br/>baitna/service.ts · baitna/prompts.ts"])
    Guard -->|no| Route{"Intent"}
    Classify --> Route

    Route -->|question| Retrieve
    Route -->|situation| Extract

    subgraph QP["Question path"]
        Retrieve["Hybrid search<br/>keyword + vector, scoped in SQL<br/>baitna/retrieve.ts · services/embedding.ts"] --> Rerank["Rerank passages<br/>free model, relevance floor<br/>baitna/rerank.ts"]
        Retrieve -->|"search failed"| Abstain
        Rerank --> Relevant{"Anything<br/>relevant?"}
        Relevant -->|no| Abstain(["Decline and say why<br/>baitna/prompts.ts"])
        Relevant -->|yes| Assemble["Assemble passages +<br/>partner's live facts<br/>baitna/assemble.ts"]
        Assemble --> Answer["Generate answer<br/>standard model<br/>baitna/generate.ts · baitna/prompts.ts"]
        Answer --> PriceQ["Hidden-price guard<br/>baitna/guards.ts"]
        PriceQ --> QOut(["Answer + sources"])
    end

    subgraph SP["Situation path"]
        Extract["Extract constraints<br/>standard model<br/>baitna/generate.ts"] --> Merge["Merge with earlier<br/>constraints in session<br/>baitna/session.ts"]
        Merge --> Thin{"Too vague, nothing<br/>from earlier?"}
        Thin -->|yes| Clarify(["Ask one question"])
        Thin -->|no| Match["Check eligibility, then match<br/>and validate listing cards<br/>baitna/eligibility.ts → baitna/app-api.ts<br/>baitna/cards.ts"]
        Match --> Any{"Any cards?"}
        Any -->|no| None(["Nothing matches + notes"])
        Any -->|yes| Intro["Write intro<br/>standard model<br/>baitna/generate.ts"]
        Intro --> Strip["Strip date, proximity<br/>and price claims<br/>baitna/guards.ts"]
        Strip --> SOut(["Intro + listing cards"])
    end

    classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef io fill:#dbeafe,stroke:#2563eb,color:#172554
    class Scope,Classify,Partners,Retrieve,Rerank,Assemble,Answer,PriceQ,Extract,Merge,Match,Intro,Strip step
    class Start,Redirect,Abstain,QOut,Clarify,None,SOut io
```

Every reply from `baitna/service.ts` is saved to conversation history (`services/conversation.ts`), including early exits and declines. Error responses from the route, such as timeouts and rate limits, are not. After a timeout the work keeps running, so the turn is still saved if it finishes late. Only the situation path turns live listings into cards. The question path uses the selected partner's current listings only as background facts, and the situation path never searches documents.

### Keeping Baitna's documents searchable

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 400}}}%%
flowchart LR
    Dashboard["Admin dashboard<br/>outside this repo"] -->|"publish / edit"| Docs[("orbit_baitna_docs")]
    Timer["Timer: every 5 min by default,<br/>first run 15 s after boot<br/>started by index.ts"] -->|"Redis lock"| Pass
    Manual["POST /orbit/baitna/reindex<br/>shared secret<br/>routes/baitna.ts"] --> Pass
    CLI["npm run index:baitna<br/>scripts/index-baitna-docs.mjs<br/>no lock, needs a build"] --> Read
    Pass["Index pass<br/>baitna/scheduler.ts"] --> Read["Read every document<br/>500 per page<br/>baitna/indexer.ts"]
    Docs --> Read
    Read --> Published{"Published?"}
    Published -->|no| Remove["Delete its chunks<br/>baitna/indexer.ts"]
    Published -->|yes| Split["Split markdown<br/>baitna/chunk.ts"]
    Split --> Changed{"Text changed?<br/>content hash"}
    Changed -->|no| Reuse["Keep existing embedding"]
    Changed -->|yes| Embed["Embed via OpenRouter<br/>services/embedding.ts"]
    Reuse --> Chunks[("orbit_baitna_chunks")]
    Embed --> Chunks
    Remove --> Chunks
    Chunks -->|"searched by"| Q["Baitna question path<br/>baitna/retrieve.ts"]

    classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef data fill:#ffe4e6,stroke:#e11d48,color:#881337
    class Pass,Read,Split,Reuse,Embed,Remove step
    class Docs,Chunks data
```

## 6. Who uses what

| Store / service | What lives there or is called | Used by |
|---|---|---|
| **Supabase** (Postgres + Auth) | Auth token check, `users` | auth middleware |
| | `match_offers_tickets_and_deals` RPC (vector search over offers, tickets, online deals) | Orbit chat, day planner |
| | `saved_plans` | save / share plan routes |
| | `analytics_events` | both chat routes |
| | `orbit_baitna_docs`, `orbit_baitna_chunks`, `orbit_baitna_search` RPC | Baitna indexer (writes), Baitna question path (reads) |
| **Redis** (in-memory fallback when `REDIS_URL` is unset, is `redis://localhost:6379`, or can't be reached) | `orbit:conversation:{user}:{session}`, 24 h, last 20 messages | Orbit chat, Baitna |
| | `orbit:plan:{id}`, 1 h | day planner writes, save route reads |
| | `limit:velocity:{user}`, `limit:daily:{user}:{date}` | rate limiter |
| | `orbit:baitna:scope:*`, `orbit:baitna:constraints:*`, 24 h | Baitna session |
| | `orbit:baitna:index:lock` | indexer timer |
| **OpenRouter** | `MODEL_FREE` | Orbit intent, offer one-liner, simple chat; Baitna classifier, reranker |
| | `MODEL_STANDARD` / `MODEL_PREMIUM` / `MODEL_VISION` | Orbit conversation, picked by `lib/model-router.ts` |
| | `MODEL_STANDARD` | Baitna answers, constraint extraction, intros |
| | `openai/text-embedding-3-small` | offer search, Baitna search, indexer |
| **Google Maps Platform** | Places (nearby, autocomplete, details) and Routes | maps routes, Orbit chat (JSON find mode), day planner |
| **Student app backend** | `/baitna/partners` (partners + listings), `/baitna/eligibility`, called with the student's own token | Baitna |
