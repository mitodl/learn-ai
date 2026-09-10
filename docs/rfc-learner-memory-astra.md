# RFC: Per-learner memory for AskTIM chatbots

## Status

Draft — not yet posted.

## Problem

AskTIM doesn't consistently remember a learner's preferences between conversations. Someone
who asked for online courses last week has to explain that again today.

Some of the information already exists. MIT Learn collects topic interests, goals,
education level, certificate preference, time commitment, and delivery preference. The
bots should be able to use those fields. Conversations can fill in things the profile
doesn't cover, such as "plain English, minimal jargon" or "advanced data science courses,
but beginner ecology."

Memory should reduce repeated questions, but it can also confuse subjects, remember a
one-time request as a permanent preference, or treat a course recommendation as a learner
interest. A mistake in memory can affect several later conversations, so this needs more
checking than a prompt change alone.

## Decision

Keep the profile and learned memory separate. MIT Learn remains the source for profile
fields; learn-ai fetches and caches them. What the bots learn from conversations stays in
learn-ai as short free-text notes: background, preferences shared across bots, and
preferences for each bot.

I'd use a free-text approach for v1. Structured facts would make individual edits and
tracking where a fact came from easier, but they would not fix a model misunderstanding
"this topic." We can revisit that if updates lose unrelated facts, preferences keep being
applied to the wrong subject or bot, or we need individual memory editing.

The Django checkpointer already stores conversations. After a reply is saved, add a work
record pointing to its checkpoint and the messages that make up that exchange. Don't copy
the conversation into another table. A background task reads those references and updates
the learner's notes, one task at a time per learner. Save the updated notes and delete the
processed work records in the same transaction. This doesn't delete the chat history.
Clearing memory discards outstanding work records and prevents earlier conversations from
being used to rebuild the notes.

This follows LangGraph's long-term memory model rather than inventing one: a profile
document per user, kept behind LangGraph's `BaseStore` interface, and written in the
background after the conversation instead of by the agent mid-turn. That is the
"profile" schema with background updates that LangChain's
[memory docs](https://docs.langchain.com/oss/python/concepts/memory) describe. Two
pieces are ours, each for a stated reason. The store is a Django-backed `BaseStore`
over a `LearnerMemoryNote` table rather than LangGraph's own `PostgresStore`, because
that one manages its tables outside Django migrations and has no user foreign key, so
"forget me" would be manual cleanup instead of a cascade delete. And extraction is one
structured-output call rather than `langmem`: langmem's trustcall patch loop does not
converge through `ChatLiteLLM`, which is how learn-ai reaches every model, and its
one-document-per-namespace shape does not fit per-bot sections. The queueing around
extraction (pending turns, a per-user lock, a clear counter) is outside what a memory
library covers; it is ordinary background-job plumbing, and it is what makes deletion
and ordering reliable. Everything runs on the
existing Django/Postgres, Redis, and Celery setup. No vector search or new memory
service for now; open to revisiting if the evaluation shows free-text notes aren't
enough.

Ship profile support first, behind a flag. Learned memory follows after the checks below.
Those checks are required before enabling it for real users, not before approving this RFC.

## Approach

### Profile

Add `GET /api/v0/profiles/<global_id>/preferences/` in mit-learn, requiring service
authentication. Return only the six fields above. The learner ID comes from the
authenticated user, never a request field. Cache by that ID for 12 hours; profile edits
may take that long to appear. Clearing the cache when a profile changes can come later.

This is a cached copy, not a second profile to keep in sync. Don't copy those fields into
the learned notes as a starting point. Also, limiting this endpoint to six fields doesn't
reduce any other access the existing service token already has.

Use a timeout on the fetch and continue without the profile if it fails. Read learned
notes separately so a profile failure doesn't hide them too. When the profile isn't
cached, the chat waits for the fetch to finish or time out. The timeout value is TBD
based on testing.

### Stored notes

Store notes under `about`, `instructions`, and `instructions:<bot>` keys. Each bot reads
the shared background and instructions plus its own instructions. Topic scope stays in
the text: "advanced data science courses" must not become "advanced courses."

Start with a 1,500-character limit per section and record when each section was updated.
Three full sections are roughly 1,100+ tokens before the profile and headings. Limit how
much profile text is included too, and measure the cost before shortening the notes.

Give the note model a user foreign key so deleting a local user deletes their notes.
Storing a user ID in the note's lookup key alone doesn't do that. An empty revised section
must clear an existing section, rather than leave it unchanged. Check that returned notes
fit the length limit instead of cutting a sentence in the middle.

Record how many times each user has cleared memory, starting at zero. This counter is
called `generation` in the database. Copy it when accepting a chat request. Before saving
the reference for processing, check that the counter hasn't changed. If the learner
cleared memory while the bot was replying, discard the exchange from memory processing.

### Tracking unprocessed exchanges and updating notes

Add `PendingMemoryTurn` with the user, generation, bot, a reference to the checkpoint the
exchange produced, a hash of that checkpoint's messages, and creation time. Each row
identifies one completed exchange awaiting memory processing: the checkpoint's last
learner message and last reply. Save it only after the checkpoint is available, and
schedule the task after the work record commits.

The worker reads that checkpoint, not whichever checkpoint is latest when the task runs.
Its last learner message is the exchange; earlier learner messages are context up to a
configured count and per-message length, with at most 500 characters of the reply. Label who said each
message. Earlier messages help explain "this topic"; the assistant's reply helps interpret
the learner's words but cannot establish facts about them.

A checkpoint isn't necessarily immutable: the
[error-recovery code](../ai_chatbots/utils.py) can rewrite its stored message list. Record
a hash of the referenced checkpoint's message data with the work record and verify it
before use. If the checkpoint has changed, has been deleted, or no longer identifies the
expected exchange, skip that work record and count the skipped update. Don't substitute
newer history. A temporary database failure should be retried, not treated as deletion.

Confirm the checkpoint and message references for each bot, including the tutor, during
implementation. References are the preferred approach if they remain usable until the
task runs. If testing shows they are too often unavailable, reconsider saving a bounded
copy of the input. For now, occasionally missing a memory update is preferable to keeping
another copy of the conversation.

The consumer already checks thread ownership. Use that validated user and thread when
saving the work record, and check ownership before reading its checkpoint. The record
should have the same access restrictions as chat history. Pass the user ID to Celery,
not the transcript.

When the first work record is saved, schedule a task to run in 15 minutes.
Exchanges arriving during that wait get their own work records for the same task; they aren't dropped and
don't restart the timer. Tune the wait during the pilot. A periodic task also finds users
with unprocessed exchanges and schedules another attempt if necessary. This recovers work
when sending a Celery task fails or a new exchange arrives just as a worker finishes.

The task acquires a Redis lock for that learner, preventing another task from updating
the same notes. It reads the oldest unprocessed exchanges, up to configured limits on
their number and total input length. Creation time and row ID determine their order.
A cheaper model first checks whether those exchanges contain anything worth remembering
for future conversations. This check is the gate. If it says yes, a stronger model revises
the notes. Both models receive the current notes and context read from the referenced
checkpoints, so
"yes, remember that" has a chance of being understood. A batch containing several bots
must keep their labels and return separate bot sections alongside the shared notes.

After validating the result, save the changed notes and delete only the selected pending
rows in one database transaction. New rows arriving during extraction stay pending. If
the gate says no, remove the selected rows without changing notes. A crash before commit
leaves the whole batch pending; a crash after commit leaves nothing from that batch to
apply again. We shouldn't rely on asking the model twice producing the same result.

Before saving the notes, lock the database row holding the learner's clear counter and
check that it still matches the task's starting value. If it changed, discard the result.
Lock that same row when adding a work record or clearing memory. Don't
hold a database transaction open during the model calls, and don't make a chat response
wait for the extraction lock just to save its pending row.

If another task holds the Redis lock, leave the work records for a later attempt.
Set limits on the batch size, input length,
model timeouts, retries, and total task runtime; the lock must last longer than the whole
task. Model timeouts alone don't bound database work or retry delays. If we can't enforce
that runtime limit, we'll need lock renewal and a check that an expired worker cannot
save. Wait longer between successive retries and report exchanges that repeatedly fail
processing. The policy for repeated
failures and retaining their work records is TBD. Skip an update when its source is no
longer usable, but don't silently drop it just because a task failed.

The queue orders turns by when their responses finish, not when requests began. That is
worth testing with overlapping requests. If we need submission order, add a turn sequence
rather than assume timestamps solve it. Prevent duplicate work records for the same
learner, generation, and exchange, using the thread and learner message ID to identify it.

Make the gate and extractor configurable, with gpt-4o-mini and gpt-4.1 as starting
candidates. Test them on batches and check in the evaluation cases before treating the
model choice as settled.

### Using memory

Label profile fields and learned notes separately. The recommendation bot gets all six
profile fields; other bots initially get education level plus the shared notes and their
own preferences. Education level doesn't tell us how well someone knows a particular
subject.

Application and tutoring rules always win. Within those rules, the current request wins
over saved preferences. Profile fields remain authoritative when they directly conflict
with older notes and the learner hasn't supplied a current override. Don't write chat
changes back to the MIT Learn profile.

A temporary request should not automatically change long-term memory. "Introductory
courses this time" can override an advanced preference for one search; "remove that
restriction" should remove it. "Regardless of difficulty" is less clear and belongs in
the evaluation cases. We need to agree on the intended behavior rather than let whichever
model we use decide the product policy.

Add an extraction rule never to remember instructions about a bot's rules, tools,
permissions, or what it may reveal. Clearly mark the notes as learner-supplied data, and
test attempts to carry a bad instruction from the recommendation bot into the tutor.
This doesn't require a closed list of allowed preferences, but a single poisoned-message
test won't be enough either.

Start by placing learner context before the static prompt, and update the prompt to use
it rather than repeat questions it answers. Appending the block may work just as well
and would help prompt caching; compare both in evaluation. For the tutor pilot, add the
learner notes at the end of the messages sent to the model, with an explicit statement
that tutoring rules win. A supported `learner_context` argument in
`open-learning-ai-tutor` remains the follow-up.

Declining a certificate doesn't tell us someone's budget. If product wants the bot not
to ask about price in that case, say
that as a policy, without implying the learner asked for free resources. Confirm with
product.

For v1, tell the bot about preferences in its prompt rather than automatically adding
filters to search requests. Applying those filters in code needs product agreement.
The prompt must explicitly tell the bot to use
relevant preferences in searches and to search before claiming there are no results.
Test the actual searches, not just prompt wording. Interests aren't automatically
exclusions, and the catalog still doesn't have a reliable beginner/advanced filter.

### Identity and scope

Everything is tied to the authenticated user's `global_id`. Request fields, cookies, and
model output cannot select another person's profile or memory. Test cache isolation,
thread authorization, and memory API access.

This depends on APISIX stripping or replacing client-supplied `x-userinfo` and preventing
untrusted access around the gateway. The middleware decodes that header; it doesn't
independently authenticate it. State and verify that deployment requirement.

Recommendation, syllabus, video, and edX tutor bots are included where that identity is
available. Anonymous chats are unchanged. Canvas stays out until its integration sends a
verified learner identity; the shared service token alone doesn't supply one.

### Clearing memory

V1 has `GET /api/v0/memory/` to inspect the notes and `DELETE /api/v0/memory/` to clear
them. Django admin can show them to authorized staff. Individual fact editing and a
separate participation setting can come later. Decide when frontend controls should ship
before broad rollout; API access is sufficient for the internal pilot.

DELETE takes the same per-user Redis lock, with a bounded wait. If it can't acquire the
lock, return a retryable failure rather than report success. In one transaction, clear
the notes and pending rows and increment the generation. This prevents both queued tasks
and responses already in progress from restoring old memory.

There is another way old facts could return: a new turn in an old thread can include them
in its history. A "context only" instruction isn't a reliable forget mechanism. After a
clear, use only the specific messages identified by work records in the new generation.
A referenced checkpoint may also contain pre-clear messages; exclude those rather than
passing its entire history to the model. If earlier work records from the new generation
have already been processed and deleted, omit that missing context rather than fall back
to untracked history. This loses
some context after a clear, but it's a small solution that doesn't require timestamps
inside checkpoint messages. If a new statement is too ambiguous without that context,
don't record it.

New explicit statements can create new memory. Clearing learned notes doesn't clear the
MIT Learn profile or ordinary chat history, and can't recall a prompt already sent to a
model. Explain those limits. Test that old facts in an assistant reply don't get recorded
as if the learner had just stated them.

Deleting a local user should also delete their notes, clear counter, and outstanding
work records. Propagating account
deletion from mit-learn is an existing gap for sessions and checkpoints too; track that
separately rather than claim this foreign key solves it.

Delete processed work records when their batch commits; leave source conversations under
the existing chat-retention rules. Memory processing must not prevent a learner from
deleting chat history. Set a retention policy for failed work records before broad
rollout, and account for personalized prompts in tracing access and
retention. Don't log whole notes or transcripts by default. Learned notes must exclude
names, emails, problem statements, attempted or correct answers, hints, grades, scores,
and problem identifiers.

## Evaluation and rollout

Ship profile support first and see whether it helps. Then test learned memory internally
before expanding bot by bot behind flags. Workers must respect write disablement, and
re-enabling should not unexpectedly process an old backlog.

Extend the existing evaluation framework with checked-in conversations across sessions.
Compare profile-only with profile-plus-memory. Required cases include:

- retaining unrelated facts through repeated rewrites, with different preferences by
  subject and bot;
- corrections, withdrawals, temporary overrides, and "yes, remember that";
- questions, complaints, and assistant suggestions that should not become learner facts;
- several poisoned instructions, including recommendation-to-tutor transfer, and no
  retained assessment content or regression in tutoring rules;
- profile conflicts, certificate policy, repeated clarification, and actual search calls;
- empty-section clearing, batches from several bots, overlapping conversations, and
  tasks delivered out of order;
- lock contention, crashes before and after commit, new rows arriving during processing,
  and recovery of stranded work;
- changed or deleted checkpoints, missing message references, duplicate work records,
  and correct selection of the learner message and reply for each bot;
- clearing during extraction, a pre-clear response finishing afterward, and new chat in
  an old thread without relearning old facts;
- user isolation, unavailable dependencies, input limits, cost, and response delay.

Use integration tests for storage, ownership, and deletion; use model runs for extraction
and answer quality. Mocked tests can check that input is passed and output is saved.
Prompt assertions check wording. Neither tells us whether a model remembers the
right things.

Report incorrect memories, missed corrections, preference adherence, repeated questions,
tutor regressions, cost, and latency, with model/prompt versions and sample sizes. Agree
on coverage and acceptance criteria before rollout. Arbitrary percentages in the RFC
wouldn't make the evidence stronger. Manual spot checks and the flag help, but don't
replace these tests.

## Alternatives considered

Structured records are still an option if free-text rewriting proves unreliable or we
need individual editing. For now I'd keep the smaller profile. LangChain describes
tradeoffs in both approaches; it doesn't settle this choice for us.

A lock and generation without pending turns would be simpler, but wouldn't preserve work
when a task skips a held lock or recover it after a crash. The pending table handles those
cases without relying on repeated model calls being harmless.
A larger event ledger or revision scheme can wait unless we add independent writers or
can't meet the lock's runtime requirement.

A plain Django model alone would also work for these fixed notes. The `BaseStore`
adapter is a thin layer over the same table, kept so that an agent-side memory tool, or
a retry of `langmem` on a later version, plugs in without a rewrite. It
doesn't handle authorization, ordering, or forgetting for us; the Django models around
it do. Start with a direct structured-output call for extraction; using LangMem instead
will require a re-test through `ChatLiteLLM` and a comparison on the same evaluation
cases.

Mem0, Zep/Graphiti, and Letta don't seem to earn their extra setup for three short notes
per turn. That may change if we need many memories, past-event retrieval, or relationships
between facts. Collections don't inherently require vector search, and retrieved facts
can coexist with preferences always included in the prompt.

Keeping memory in mit-learn could simplify ownership, but learn-ai already has the source
chats and workers. It would also require a write API. Keep it here for now, with the
cross-service deletion dependency acknowledged.

## Consequences

The design adds stored notes, a table identifying exchanges waiting to be processed, and
a counter that prevents cleared memory from being restored by older tasks. Conversation
text remains in the checkpointer. No new service is required.

There is extra prompt cost, some profile-fetch delay on cache misses, and a gate plus
rewrite cost per accepted batch. The wait before processing means new memory isn't immediately
available. An exchange is skipped if its referenced checkpoint has changed or been
deleted. Clearing memory deliberately reduces the history available to extraction.

Whole-note rewriting and prompt-only search remain the main quality doubts. The checks
above should tell us whether they are good enough for v1 or whether we need to reconsider.

## Open questions

- Who owns the replay fixtures, evaluation review, and rollout criteria?
- How long should we wait before processing exchanges, how many should a task read, and
  what fetch and task timeouts work in practice? TBD.
- Should declining a certificate suppress price questions?
- Should profile preferences become search filters, and how should overrides work?
- When do frontend controls and a separate memory participation setting ship?
- How long do we keep failed pending turns, and how do we handle repeated failures?
- Do each bot's saved checkpoints and message IDs remain usable long enough for extraction,
  and how often do changed or missing sources cause an update to be skipped?
- Which issue tracks cross-service account deletion, and when can the tutor expose a
  supported learner-context argument?

## References

- [LangChain memory overview](https://docs.langchain.com/oss/python/concepts/memory)
- [Celery locking example](https://docs.celeryq.dev/en/stable/tutorials/task-cookbook.html)
- [Django transactions](https://docs.djangoproject.com/en/5.2/topics/db/transactions/)
