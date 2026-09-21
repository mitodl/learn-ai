# RFC: Per-learner memory for AskTIM chatbots

## Status

Draft. Working prototype on the
[`mb/learner-memory-poc`](https://github.com/mitodl/learn-ai/tree/mb/learner-memory-poc)
branch of learn-ai. Implementation details will be spelled out in a separate spec before
implementation begins.

## Problem

AskTIM doesn't remember a learner's preferences between conversations. Someone who asked
for online courses last week has to explain that again today.

Some of the information already exists. MIT Learn collects topic interests, goals,
education level, certificate preference, time commitment, and delivery preference when a
learner fills out their profile. Conversations can fill in what the profile doesn't cover,
such as "plain English, minimal jargon" or "advanced data science courses, but beginner
ecology."

Memory can also go wrong in ways a prompt change can't: confusing two subjects,
remembering a one-time request as a permanent preference, or treating a course the bot
recommended as something the learner wants. A mistake in memory carries into every later
conversation, so this needs more checking than a prompt change.

## General Approach

Give AskTIM the learner's MIT Learn profile fields, short notes learned from
conversations, and eventually their enrollments, so preferences carry across sessions and
bots.

- **Profile fields.** Topic interests, goals, education level, certificate preference,
  time commitment, and delivery preference (in-person, hybrid, or online) stay in
  mit-learn. AskTIM reads them but does not change them or copy them into learned notes.
  This also applies when a profile field is empty: learners save these preferences on
  their MIT Learn profile, not through chat.
- **Learned notes.** An **About** note holds additional background, such as occupation; a
  **Preferences** note holds response preferences such as tone, length, and level of
  jargon. All included bots read those two notes. Recommendation, syllabus, and video
  bots also have their own **Per-bot preferences** note, such as course exclusions for
  the recommendation bot. The tutor only reads the shared notes in v1. A background task
  gathers completed exchanges (a learner message and the bot's reply) and asks a model
  to update the notes. Initially, it runs 15 minutes after the first exchange waiting to
  be processed; further exchanges during that window join the same run. I call the step
  that identifies and saves useful information _extraction_ below.
- **Enrollments (phase 2).** Read from an enrollment data source, never written into
  notes. Where to get the data is open; see Phase 2.

An application-wide feature flag controls availability: on in the release-candidate (RC)
environment, off in production until the evaluation passes. Separately, each learner has
a personalization on/off setting that controls whether AskTIM uses their data.
Learners can see what has been remembered, clear it, and turn personalization off from
their MIT Learn settings page; off covers everything here.

## Learner Memory Storage — Options Considered

Both options remember the same kinds of facts and preferences from conversations and use
the existing profile fields. The choice is how to organize and update learned memory:
rewrite a few notes or update individual items. Both use the bot scopes described above.
Learned memory stays in learn-ai, which already has the
conversations and background workers; mit-learn would need a write API.

### Option 1: Free-text notes (recommended)

The About and Preferences notes, plus each bot's own preferences note, are rewritten by
a background model call: the "profile document with background updates" pattern from LangChain's
[memory docs](https://docs.langchain.com/oss/python/concepts/memory).

- Pro: one revision updates the notes together, without identifying and updating
  individual items. The notes can go directly into the prompt as three short sections,
  each with a size limit to bound prompt cost.
- Con: whole-note rewriting can drop unrelated facts or over-generalize a topic-specific
  preference. This is the main quality risk and needs a real evaluation set. Learners can
  correct a note by saying so in a chat, but can't edit or delete one item; clear is all
  or nothing.

### Option 2: Individual memory items

Store each preference as its own row and have the extraction model emit adds, updates,
and deletes. The rows could be structured (`subject`, `predicate`, `value`, `source`,
`confidence`) or just short text with a bot scope, a source, and a timestamp.

- Pro: items are easy to edit, delete, and trace; one rewrite can't lose an unrelated
  item.
- Con: doesn't fix the harder problem, the model misreading "this topic" or a one-time
  request; the mistake lands in a row instead of a sentence. A structured schema for
  open-ended preferences is hard to get right up front. The short-text variant avoids
  that but still needs the model to emit per-item operations rather than one revision,
  and it is more code and tests for the same v1 outcome.

## Shared Implementation Choices

Either storage format uses the existing Django/Postgres, Redis, and Celery stack. The
model sees the conversation and existing memory together so it can interpret responses
such as "yes, remember that." Updates run after the reply, so an incorrect memory cannot
affect the reply that triggered it.

**Extraction.** I tried `langmem`, LangChain's memory extraction package, first. It
didn't reliably return results through `ChatLiteLLM`, which is how learn-ai reaches every
model, and its last release was 2025-10-27. A direct structured-output call is about
thirty lines and easier to steer; the cost is that we own the prompt and its evaluation.

**Storage backend.** LangGraph's `PostgresStore` lives outside Django migrations and has no user
foreign key. External memory services (Mem0, Zep, Letta) add vector retrieval we'd want
for hundreds of memories per learner, not a few short notes, plus their own
authentication, deletion, and data-residency questions. A thin Django-backed store over
our own table keeps LangGraph's `BaseStore` interface, gives us migrations and cascade
deletes, and can be swapped for either later without touching the bots.

## Decision

**Option 1: free-text notes**, behind a feature flag: RC first, production once the
evaluation passes.

Short notes let us evaluate conversational memory before investing in individual items;
if rewrites keep losing facts, or product needs item-level editing, Option 2 is the next
step. That would require changes to extraction, storage, and learner controls, while
keeping the same source of profile data. No vector search or third-party memory service
in v1.

Recommendation, syllabus, and video bots see three note sections: **About** (additional
background not covered by the profile), **Preferences**
(how they want every bot to respond), and **Per-bot preferences** (such as "don't show me
MicroMasters programs" for the recommendation bot). After a couple of conversations a
learner's notes might read:

> **About:** Works as a data analyst.
> **Preferences:** Plain English, minimal jargon, short answers.
> **Recommendation bot:** Advanced data science courses; beginner ecology courses.

If their MIT Learn profile specifies online courses, the next time they ask "anything
good on ecology?", the recommendation bot would combine that profile preference with the
beginner ecology level in its own note and the short-answer preference in the shared
note. It would look for online, beginner-friendly ecology courses and answer briefly.
The online preference comes from the profile, not from a learned note.

Some things must not be written to memory: names, email addresses, assessment content
(problem statements, answers, hints, grades, problem identifiers), and anything about how
a bot should treat its own rules, tools, or permissions, so "ignore your rules from now
on" shouldn't stick. Tutor transcripts are never sent to extraction in v1; the tutor reads
the shared notes but doesn't write them, which keeps the main source of assessment content
out of the pipeline. Everything else relies on the extraction prompt and the evaluation
cases. That is a tested mitigation, not a guarantee: the extraction model still sees
whatever the learner typed.

v1 makes one extraction call per batch. The prototype also has a cheaper screening call
that decides whether a batch is worth extracting; I've taken it out of v1 because it adds
a second chance to miss a correction, and on RC the saving is negligible. If extraction
cost matters in production, screening can be reconsidered behind the same evaluation.

In one prototype run it extracted three reasonable preferences from four exchanges,
dropped a planted "ignore your rules" message, and carried the preferences into a new
browser session. Two failures shaped the design and are now evaluation cases: an early
version generalized "advanced data science courses" into "advanced everything", and the
bot sometimes repeated a preference without applying it to search.

## Approach

- **mit-learn: preferences endpoint.** `GET /api/v0/profiles/<global_id>/preferences/`
  returns the six profile fields only, no name or email. learn-ai authenticates with the
  `LEARN_ACCESS_TOKEN` it already uses for content-file search. This identifies learn-ai's
  service account, which receives a new group permission to read any learner's six
  profile fields; the `global_id` in the URL selects the learner. Ordinary signed-in
  learners do not receive that permission. learn-ai must select the ID from the
  authenticated chat user, not a caller-supplied ID. A leaked service token could expose
  these fields across learners; limiting the response excludes names and email addresses
  but does not restrict the token to one learner.
- **learn-ai: profile fetch and cache.** Fetched only for learners with personalization
  on, keyed on the gateway-authenticated `global_id`, cached for 12 hours to start. A
  failed fetch leaves the chat running without the profile.
- **learn-ai: notes and extraction.** learn-ai stores three things: each learner's
  personalization on/off setting, their learned notes, and a list of completed chat
  exchanges waiting to be checked for information worth remembering. These pending
  exchanges point to saved conversations rather than making another copy of their text.
  A background worker processes several exchanges from the same learner together and
  updates the notes. Only one worker can update a learner's notes at a time. Saving the
  notes and removing the processed work happen together: if that save fails, the work
  remains available for retry; if it succeeds, the work is finished.
  The pending-work table preserves unfinished work when a worker crashes or another
  worker is already busy with that learner. A lock prevents simultaneous updates but
  does not itself remember work to retry; the table and a periodic retry task provide
  that recovery.
- **learn-ai: learner controls.** One endpoint for the signed-in learner: read the notes
  and setting, turn personalization off (deletes notes and queued work in the same
  transaction), or clear notes while leaving it on. The MIT Learn settings page gets a
  section that calls it. Anonymous and Canvas requests do not provide learn-ai with a
  trusted individual learner identity, so they have no memory and no controls.
  These controls must be available before or when memory is enabled for learners.
- **Prompt assembly.** Profile fields and notes are labeled as learner-supplied data, not
  instructions. Application and tutoring rules always win; what the learner says now
  takes precedence for the current conversation. A request such as "in-person this time"
  can override the profile for that conversation without changing the profile or being
  saved in a note. Likewise, "regardless of difficulty" overrides a saved level preference
  for that request only; it does not remove the preference. Notes should not duplicate
  profile fields; if a conflicting value
  nevertheless appears in a note, the profile wins. The recommendation bot receives all
  six profile fields; other bots receive education level. The bot is told the preferences
  and decides how to search. We don't turn them into search filters in code, because the
  catalog has no reliable level filter and that needs product agreement.
- **Tutor.** For the pilot, learn-ai appends the shared notes to the tutor's messages
  with an explicit statement that tutoring rules win. The tutor cannot yet place these
  notes in a dedicated part of its own prompt, so this path needs particular testing for
  saved instructions that try to override tutoring rules. A supported `learner_context`
  argument in `open-learning-ai-tutor` is the proper fix and is a follow-up.

### What "off" and "clear" mean

| Control      | Effect                                                                                                                                                                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Turn off     | New chats load no profile or notes and queue no work. Existing notes and queued exchanges are deleted. A chat already being answered finishes with the context it started with.                                                                   |
| Clear        | Learned notes and queued memory work are deleted; personalization stays on. Profile fields remain in mit-learn and can still be used. Clear wins over an extraction already running: either the extraction aborts or clear removes what it wrote. |
| Chat history | Untouched by both. A later message in the same thread carries earlier messages as context, so a cleared fact can be learned again; deleting the chat is the existing control for that.                                                            |

### Evaluation

Use a combination of manual testing by ODL staff and automated scoring through learn-ai's
evaluation management command to assess how learner memory improves or degrades answers.
Compare answers with and without learner memory across conversations, and use the results
to decide whether it is ready for production.

## Phase 2: Enrollment History

The bots should eventually know which courses a learner is enrolled in or has taken,
mainly so the recommendation bot stops suggesting them. This needs more discussion, so it
isn't in v1.

The awkward part is where the data lives: mit-learn stores no enrollments, the dashboard
reads them from MITx Online with the learner's own session, and the warehouse table that
has them isn't served anywhere. The routes I see:

- **A.** Warehouse view pulled into mit-learn by the existing certificate ETL. Every
  platform; blocked on confirming the lookup through `dim_user` returns all platform
  accounts belonging to the learner and none belonging to anyone else.
- **B.** New MITx Online service endpoint by `global_id`. Live, MITx Online only.
- **C.** Serve `ProgramCertificate` as is. Completed programs only.
- **D.** The AskTIM drawer sends enrollments with each message. These would be
  browser-supplied claims, not verified enrollment records. They could not establish
  identity or authorize access, and this route would not cover tutor or Canvas chats.

I lean towards A if this becomes mandatory, or B if MITx Online-only is acceptable. Very
open to other ideas.

## Consequences

**What we gain:** bots stop re-asking what the profile already says, and carry
conversational preferences (tone, level by subject, exclusions) across sessions and bots.

**Risks and costs:** whole-note rewriting and relying on the prompt rather than search
filters are the main quality doubts; the evaluation decides whether they are good enough.
Memory is not free: roughly 1,100 extra prompt tokens per request with full notes, a
profile fetch on cache misses, and an extraction call per learner per batch. Notes usually
update after the 15-minute batching window; retries can take longer. Profile edits can take
up to 12 hours to appear. Retention is a new policy: notes not updated for 365 days are
deleted, while signed-in chats
aren't expired today, and account deletion doesn't propagate from mit-learn to learn-ai,
so the staff deletion process must be verified to cover learn-ai before production.

**Required changes in other applications:** mit-learn gets the preferences endpoint, its
permission, and the settings section; open-learning-ai-tutor gets `learner_context`, as a
follow-up after the pilot.

## Open Questions

- **Default on or off:** This RFC assumes on by default with an opt-out. The alternative
  is off by default with a one-time "remember my preferences?" prompt in the chat drawer,
  which would double as the disclosure. A further step is per-item consent ("remember
  this"), which makes persistence intentional but puts the burden back on the learner.
  MIT Learn has learners who list secondary school as their education level, which argues
  for asking first. Blocking for rollout.
- **Item-level editing:** Is allowing a user to clear all memories good enough, or should
  learners be able to remove individual memories and keep others?
- **Enrollment history:** which phase 2 route, if the requirement becomes mandatory.
  Non-blocking.
- **Cross-service deletion:** how should account deletion in mit-learn reach learn-ai
  automatically? Automation is non-blocking; the staff process covers deletion until then.
- **Memory in traces:** learner notes will appear in PostHog and Opik traces of
  personalized prompts. Only staff can access those, but a data breach there would expose
  them. Is that a concern? Excluding them would make it harder to debug why the AI
  answered the way it did.
