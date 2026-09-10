# RFC: Per-learner memory for AskTIM chatbots

**Authors:**

| Name          | GitHub User |
| ------------- | ----------- |
| Matt Bertrand | @mbertrand  |

**Scope**

- How AskTIM bots remember a learner's background and preferences between conversations,
  and how learners can see and clear what was remembered.

## Status

Draft — not yet posted. Working prototype on the
[`mb/learner-memory-poc`](https://github.com/mitodl/learn-ai/tree/mb/learner-memory-poc)
branch of learn-ai.

## Problem

AskTIM doesn't remember a learner's preferences between conversations. Someone who asked
for online courses last week has to explain that again today.

Some of the information already exists. MIT Learn collects topic interests, goals,
education level, certificate preference, time commitment, and delivery preference when a
learner fills out their profile. The bots should use those fields. Conversations can fill
in things the profile doesn't cover, such as "plain English, minimal jargon" or "advanced
data science courses, but beginner ecology."

Memory should reduce repeated questions, but it can also go wrong in ways a prompt change
can't: confusing two subjects, remembering a one-time request as a permanent preference,
or treating a course the bot recommended as something the learner is interested in. A
mistake in memory carries into every later conversation, so this needs more checking than
a prompt change alone.

## Decision

Keep two kinds of memory separate. The MIT Learn profile stays the source of truth for the
six profile fields; learn-ai fetches and caches a copy. What the bots learn from
conversations is stored in learn-ai as three short free-text notes per learner:

- **About**: durable facts about the person (background, occupation, goals, time available).
- **Instructions**: how they want every bot to behave (tone, length, level of jargon).
- **Per-bot instructions**: how they want a particular bot to behave, such as "don't show
  me MicroMasters programs" for the recommendation bot.

For example, after a couple of conversations a learner's notes might read:

> **About:** Works as a data analyst; wants to move into ecology or environmental science.
> Prefers online courses.
> **Instructions:** Plain English, minimal jargon, short answers.
> **Recommendation bot:** Advanced data science courses; beginner-level for ecology.

The next time they open the recommendation bot and ask "anything good on remote sensing?",
the bot would search for online, beginner-friendly ecology-adjacent courses and answer
briefly, without asking about their level or delivery preference first. Today it asks.

Each bot reads the profile and notes at the start of every conversation. Updating the notes
happens in the background, about 15 minutes after a reply, not while the learner is
waiting. A cheap model first decides whether the conversation contained anything worth
remembering; most turns don't. If it did, a stronger model rewrites the notes with the
conversation in front of it. Learners can view and clear their notes through the API.
Clearing is designed so that a background update still in flight can't quietly restore
what was just deleted.

Technically this follows LangGraph's long-term memory model rather than inventing one: a
profile document per user, kept behind LangGraph's `BaseStore` interface, written in the
background after the conversation rather than by the agent mid-turn. The store
implementation is ours, a thin `DjangoMemoryStore` over a `LearnerMemoryNote` table, and so
is the extraction call, one structured-output request through `ChatLiteLLM`. Everything
runs on the existing Django/Postgres, Redis, and Celery stack; no vector search and no new
memory service. The Alternatives section says why not `PostgresStore` or `langmem`.

I'd use free-text notes for v1 rather than a table of individual facts. Structured facts
would make single-fact editing and "where did this come from" easier, but they wouldn't fix
the harder problem, which is a model misreading "this topic". We can revisit if rewrites
keep losing unrelated facts or if we need per-fact editing.

Some things are never written to memory: names, email addresses, and anything from
tutoring sessions that looks like assessment content (problem statements, attempted or
correct answers, hints, grades, scores, problem identifiers). Instructions about how a bot
should behave toward its own rules, tools, or permissions are also never remembered, so a
message like "ignore your rules from now on" in one bot can't be carried into another.

Ship profile support first, behind a feature flag. Learned memory follows after the
evaluation described below. That evaluation is a gate on enabling it for real users, not
on approving this RFC.

What I've observed so far: the prototype runs end to end locally. In one test run with
four queued exchanges, it extracted three reasonable preferences and dropped a planted
"ignore your rules" message. Preferences stated in one thread showed up in a new thread and
a new browser session. It also surfaced two real quality problems that shaped the design:
an early version generalised "advanced data science" into "advanced everything", and the
bot sometimes read a preference back without actually applying it to the search. Both are
now evaluation cases.

## Approach

**Profile.** Add `GET /api/v0/profiles/<global_id>/preferences/` to mit-learn, returning
only the six profile fields (no name, email, or avatar, which limits what a leaked token
could expose). learn-ai calls it with the `global_id` of the authenticated learner, never an
ID from the request body, and caches the result for 12 hours, so a profile edit can take up
to that long to reach the bots. The fetch has a timeout, and if it fails the chat continues
without the profile. Learned notes are read separately so a profile failure doesn't hide
them too. The prototype uses a stub for this endpoint; it is the one piece of work outside
learn-ai.

Authentication reuses what learn-ai already does for content-file and learning-resource
search: every server-side call to mit-learn carries `Authorization: Bearer
<LEARN_ACCESS_TOKEN>` ([utils.py](../ai_chatbots/utils.py)). Reading the code, mit-learn's
Django never looks at that header. The token is validated at the mit-learn APISIX gateway,
whose openid-connect plugin checks bearer tokens against Keycloak and passes the resulting
identity to Django in the `x-userinfo` header, where the existing `ApisixUserMiddleware`
resolves it to a User. So from mit-learn's point of view, learn-ai is one particular
Keycloak-backed user account, and `LEARN_ACCESS_TOKEN` is that account's token. I'm not
sure how that token is issued or rotated today (the learn-ai Keycloak client has service
accounts disabled, and the README says to copy the value from the RC pod); that's worth
pinning down with DevOps as part of this work.

The new endpoint has to be restricted to that service identity, not to any logged-in
user, or a browser session could read other learners' preferences by guessing a
`global_id`. The simplest option is a DRF permission class in mit-learn that allows only
users in a designated group or an allowlist setting of service `global_id`s, and learn-ai's
account goes in it. The alternative is an APISIX key-auth consumer for learn-ai, the way
learn-ai itself authenticates the Canvas plugin with `canvas_token`, which would keep the
endpoint off the user-facing OIDC route entirely. I lean toward the permission class since
it needs no gateway change, but either works. Limiting this endpoint to six fields doesn't
reduce any other access the existing token already has.

**Notes.** Three new Django models in `ai_chatbots/models.py`:

- `LearnerMemoryNote`: one row per user and key (`about`, `instructions`,
  `instructions:<bot>`), with the text and a last-updated timestamp. A foreign key to the
  user gives cascade deletes.
- `LearnerMemoryState`: one row per user holding a `generation` counter, incremented each
  time the learner clears memory, and a `cleared_at` timestamp. This is what stops a stale
  background update from restoring cleared notes.
- `PendingMemoryTurn`: one row per exchange waiting to be processed, described below.

`DjangoMemoryStore` in `ai_chatbots/memorystores.py` exposes the notes through LangGraph's
`BaseStore` interface, namespaced by the learner's `global_id`. It's a thin adapter over
the same table, so an agent-side memory tool or a later retry of `langmem` can plug in
without a rewrite. Authorization, ordering, and forgetting are handled by the Django models
around it, not by the store.

Each section is capped at `AI_MEMORY_MAX_CHARS` (1,500) characters. Three full sections add
roughly a page of text, on the order of 1,100 tokens, to every request. I want to measure
that cost during the pilot before deciding whether to shorten them. Topic scope has to
survive in the text: "advanced data science courses" must not become "advanced courses".

**Updating the notes.** The `AsyncDjangoCheckpointer` already saves every conversation as
`DjangoCheckpoint` rows (a checkpoint is the saved state of a thread after each reply).
Rather than copy conversation text into another table, the consumer saves a
`PendingMemoryTurn` after the reply's checkpoint is committed: the user, bot, generation,
a foreign key to that checkpoint, and a hash of its messages. Each row means "this
exchange (the checkpoint's last learner message plus the bot's reply) is waiting to be
looked at."

The first pending row for a learner schedules the `process_learner_memory` Celery task
`AI_MEMORY_DELAY_SECONDS` (15 minutes) later; rows arriving during that window join the
same run. A periodic `requeue_stale_memory_turns` task picks up anything left behind. The
task takes a per-learner Redis lock, reads the oldest pending rows up to a batch limit,
and renders each referenced checkpoint into labelled learner/assistant text with a few
earlier messages for context. A cheap gate model (`AI_MEMORY_GATE_MODEL`, gpt-4o-mini to
start) answers yes or no on whether there's anything worth remembering. If yes, the
extraction model (`AI_MEMORY_EXTRACTION_MODEL`, gpt-4.1 to start) receives the current
notes plus the rendered exchanges and returns a `MemoryRevision`, a pydantic structured
output with the three sections. After validation, the notes are saved and the processed
rows deleted in one transaction. Both models are configurable and should be compared on
the evaluation cases before we treat the choice as settled.

The details that make this safe under concurrent chats, crashes, and "clear memory"
(generation checks, hash verification, lock lifetimes, retry policy) are in the collapsed
section at the end for anyone reviewing the code.

**Using memory in the prompt.** Profile fields and learned notes are labelled separately
and clearly marked as learner-supplied data, not instructions to the bot. The
recommendation bot gets all six profile fields; other bots initially get education level
plus the shared notes and their own section. Application and tutoring rules always win.
Within those, what the learner says now wins over what they said before, and the profile
wins over an older note when they directly conflict. Nothing is written back to the MIT
Learn profile.

For v1 the bot is told about preferences in its prompt and decides how to search;
we don't automatically turn preferences into search filters in code. That would need
product agreement, and the catalog still doesn't have a reliable beginner/advanced
filter. The prompt has to say explicitly to use relevant preferences in searches and to
search before claiming nothing matches, because in testing the bot sometimes
acknowledged a preference and then ignored it.

For the tutor pilot, the notes are appended to the messages sent to the model with an
explicit statement that tutoring rules win. A proper `learner_context` argument in
`open-learning-ai-tutor` is a follow-up.

**Identity.** Everything is keyed on the authenticated user's MIT Learn ID. Request
fields, cookies, and model output can't select someone else's profile or memory. This
relies on the API gateway (APISIX) to tell learn-ai who the logged-in user is; learn-ai
trusts that header rather than authenticating it independently. If that gateway setting
were wrong, one person's memory could be shown to another, so confirming it is part of
rollout. Recommendation, syllabus, video, and edX tutor bots are included where that
identity is available. Anonymous chats are unchanged. Canvas stays out until its
integration sends a verified learner identity; the shared service token alone doesn't.

**Viewing and clearing.** `LearnerMemoryView` in learn-ai serves `GET /api/v0/memory/` to
see the notes and `DELETE /api/v0/memory/` to clear them; Django admin shows them to staff.
DELETE takes the same per-learner lock with a short bounded wait, and in one transaction
clears the notes, deletes pending rows, and increments the generation, so neither a queued
task nor a response already in progress can bring the old notes back. Frontend controls,
per-fact editing, and a separate "don't remember me" setting can come later; API access is
enough for the internal pilot, but we should decide when the UI ships before broad
rollout. Clearing learned notes doesn't clear the MIT Learn profile or chat history, and
can't recall a prompt already sent to a model. We should explain those limits to learners.
New explicit statements after a clear create new memory; old statements don't come back.

Deleting a local learn-ai user deletes their notes and pending work. Account deletion
propagating from mit-learn is an existing gap for chat sessions too; that's tracked
separately and this doesn't fix it.

## Evaluation and rollout

Ship profile support first and see whether it helps. Then test learned memory internally
before expanding bot by bot behind flags.

Extend the existing evaluation framework with checked-in multi-session conversations, and
compare profile-only against profile-plus-memory. The behaviour cases I think are
required:

- unrelated facts survive repeated rewrites, with different preferences by subject and bot;
- corrections, withdrawals, temporary overrides ("introductory this time"), and "yes,
  remember that";
- questions, complaints, and the bot's own suggestions don't become learner facts;
- planted instructions, including one that tries to travel from the recommendation bot
  into the tutor, and no assessment content retained;
- profile conflicts, the certificate/price policy, and the bot actually calling search
  with the preference rather than just mentioning it;
- clearing memory mid-conversation, and chatting again in an old thread without the old
  facts coming back.

Storage, ownership, and deletion get integration tests. Extraction and answer quality
need real model runs; a mocked test can show input went in and output was saved, but not
whether the model remembered the right thing.

Report incorrect memories, missed corrections, preference adherence, repeated questions,
tutor regressions, cost, and latency, with model and prompt versions and sample sizes. We
should agree on acceptance criteria before rollout.

Two policy questions need a product answer rather than whatever the model decides:
whether "regardless of difficulty" removes a saved level preference or only overrides it
once, and whether declining a certificate should stop the bot asking about price
(declining a certificate doesn't tell us someone's budget).

## Alternatives considered

**Structured facts instead of free text.** Still an option if rewriting proves unreliable
or we need per-fact editing. For now I'd keep the smaller design.

**Memory libraries.** This follows the "profile with background updates" pattern from
LangChain's memory docs. Two pieces are ours, each for a reason. The store is
`DjangoMemoryStore` rather than LangGraph's `PostgresStore`, because `PostgresStore`
manages its tables outside Django migrations and has no user foreign key, so "forget me"
would be manual cleanup instead of a cascade delete. And extraction is one
structured-output call rather than `langmem`: I tried `langmem` first, and its trustcall
patch loop didn't converge reliably through `ChatLiteLLM`, which is how learn-ai reaches
every model; its one-document-per-namespace shape also doesn't fit per-bot sections.
Retrying `langmem` on a later version would need a re-test through `ChatLiteLLM` and a
comparison on the same evaluation cases. The queueing around extraction (pending turns,
per-user lock, generation counter) is outside what any memory library covers; it's
ordinary background-job plumbing. Mem0, Zep/Graphiti, and Letta don't seem to earn their
setup for three short notes per learner. That may change if we need many memories,
past-event retrieval, or relationships between facts.

**Simpler queueing.** A per-learner lock and generation counter without `PendingMemoryTurn`
would be simpler, but would lose updates whenever a task skipped a held lock or crashed.
The pending table costs one small model and handles both without assuming a repeated model
call is harmless. A plain Django model without the `BaseStore` adapter would also work for
three fixed notes; the adapter is kept for the plug-in reason above. A larger event ledger
or revision scheme can wait unless we add independent writers.

**Keeping memory in mit-learn.** Would simplify ownership, but learn-ai already has the
conversations and the background workers, and it would need a write API. Keep it here,
with the cross-service deletion dependency acknowledged.

## Consequences

Adds three small tables (`LearnerMemoryNote`, `LearnerMemoryState`, `PendingMemoryTurn`),
a `BaseStore` adapter, two Celery tasks, and one API view. No new service; everything runs
on the existing Django/Postgres, Redis, and Celery setup. Conversation text stays in the
checkpointer.

Costs: extra prompt tokens on every request, a short profile-fetch delay on cache misses,
and a gate call plus an occasional rewrite per learner per batch. New memory isn't
available for about 15 minutes after a conversation, and profile edits can take up to 12
hours to appear. Occasionally an exchange is skipped because the conversation it points to
was changed or deleted; I think that's better than keeping another copy of the transcript.

Work outside learn-ai: the mit-learn profile endpoint, a `learner_context` argument in the
tutor repo, and eventually frontend controls.

The main quality doubts are whole-note rewriting and prompt-only search. The evaluation
should tell us whether they are good enough for v1.

## Open questions

- Who owns the evaluation conversations, review, and rollout criteria?
- How is `LEARN_ACCESS_TOKEN` issued and rotated, and should the profile endpoint be gated
  by a mit-learn permission class or an APISIX consumer?
- Delay before processing, batch size, and fetch/task timeouts: all tunable, none tuned yet.
- Should declining a certificate suppress price questions? Should "regardless of
  difficulty" remove a saved preference?
- Should profile preferences become search filters in code, and how do overrides work?
- When do frontend controls and a "don't remember me" setting ship?
- How long do we keep exchanges that repeatedly fail processing?
- Which issue tracks cross-service account deletion, and when can the tutor expose a
  supported learner-context argument?

## References

- [LangChain memory overview](https://docs.langchain.com/oss/python/concepts/memory)
- [Celery locking example](https://docs.celeryq.dev/en/stable/tutorials/task-cookbook.html)
- [Django transactions](https://docs.djangoproject.com/en/5.2/topics/db/transactions/)

<details>
<summary>Reliability details for code reviewers</summary>

**Pending work records.** `PendingMemoryTurn` holds the user, generation, bot, a foreign
key to the checkpoint the exchange produced, a hash of that checkpoint's messages, and a
creation time. It is saved only after the checkpoint exists, using the user and thread the
consumer already validated, and the Celery task is scheduled after the row commits. The
task receives the user ID, never the transcript. Duplicates for the same learner,
generation, and exchange are prevented using the thread and learner message ID.

**Reading the exchange.** The worker reads the referenced checkpoint, not whichever is
latest. Its last learner message is the exchange; earlier learner messages are context up
to a configured count and per-message length, with at most 500 characters of the reply,
each labelled by speaker. The reply helps interpret the learner's words but can't
establish facts about them. A checkpoint isn't immutable: the error-recovery code in
`ai_chatbots/utils.py` can rewrite its message list. The worker verifies the stored hash
before use. If the checkpoint changed, was deleted, or no longer identifies the expected
exchange, the row is skipped and counted, with no fallback to newer history. A transient
database error is retried, not treated as deletion. If testing shows references are too
often unusable by the time the task runs, reconsider storing a bounded copy of the input.

**Scheduling.** The first pending row for a learner schedules a task in
`AI_MEMORY_DELAY_SECONDS` (default 900). Later rows during that window join the same task
and don't restart the timer. A periodic task (`requeue_stale_memory_turns`) finds learners
with unprocessed rows and reschedules, which covers a lost Celery send or a row that
arrived just as a worker finished.

**The task.** Acquire a per-learner Redis lock (`AI_MEMORY_LOCK_SECONDS`, longer than
`AI_MEMORY_TASK_TIME_LIMIT`). If it's held, leave the rows for a later attempt. Read the
oldest rows up to `AI_MEMORY_BATCH_SIZE` and `AI_MEMORY_BATCH_CHARS`, ordered by creation
time then ID. Run the gate model, then the extraction model, both with the current notes
and the checkpoint context. A batch spanning several bots keeps their labels and returns
separate per-bot sections. Don't hold a database transaction open across the model calls.
After validating the result (an empty section clears the existing one; a section that
doesn't fit the cap on a clean boundary keeps the previous text rather than truncating
mid-sentence), lock the learner's generation row, check it still matches the value read at
task start, then save the notes and delete only the selected rows in one transaction. If
the generation changed, discard the result. A gate "no" deletes the selected rows without
touching the notes. A crash before commit leaves the batch pending; a crash after leaves
nothing to redo. Rows that repeatedly fail (`AI_MEMORY_MAX_ATTEMPTS`) are reported; their
retention policy is TBD.

**Ordering.** Rows are ordered by when responses finished, not when requests began. Worth
testing with overlapping requests; if submission order matters, add a sequence rather than
trusting timestamps.

**Clearing.** `DELETE /api/v0/memory/` takes the same Redis lock with a bounded wait
(`AI_MEMORY_CLEAR_WAIT_SECONDS`) and returns a retryable failure if it can't get it. In one
transaction it clears the notes, deletes pending rows, and increments the generation. A
response already in progress copies the generation at request time and checks it hasn't
changed before saving its pending row. After a clear, extraction uses only the messages
identified by pending rows in the new generation; pre-clear messages in the same
checkpoint are excluded rather than passed as history. If a new statement is too ambiguous
without that context, it isn't recorded.

**Retention and logging.** Processed rows are deleted when their batch commits; source
conversations stay under existing chat-retention rules, and memory processing must not
block a learner from deleting chat history. Workers respect the write-disable flag, and
re-enabling shouldn't process an old backlog unexpectedly. Don't log whole notes or
transcripts by default; account for personalized prompts in tracing access and retention.

**Additional cases the evaluation and integration tests should cover:** empty-section
clearing, batches from several bots, tasks delivered out of order, lock contention,
crashes before and after commit, rows arriving during processing, recovery of stranded
work, changed or deleted checkpoints, missing message references, duplicate rows, correct
selection of the learner message and reply for each bot (including the tutor), a pre-clear
response finishing after a clear, user isolation, unavailable dependencies, and input
limits.

</details>
