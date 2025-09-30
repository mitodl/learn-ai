Release Notes
=============

Version 0.16.3
--------------

- update open-learning-ai-tutor (#321)
- API endpoint for rating AI responses (#312)
- feat: Updated pyproject.toml to install Granian (#318)

Version 0.16.2 (Released September 25, 2025)
--------------

- temporarily limit canvas response keys (#313)
- security: disable yarn postinstall scripts (#317)

Version 0.16.1 (Released September 25, 2025)
--------------

- Fix chatbot session keys for anon users (#314)
- add created_at to TutorBotOutput and DjangoCheckpoint (#311)
- Associate TutorBotOutputs with UserChatSessions (#306)

Version 0.16.0 (Released September 22, 2025)
--------------

- Tweak  checkpoint data for TutorBotOutput (#309)

Version 0.15.2 (Released September 18, 2025)
--------------

- Adjust metadata writes (#307)
- Standardize checkpoints for all bots (#301)

Version 0.15.0 (Released September 16, 2025)
--------------

- Make sentry more AI-friendly (#302)
- Update dependency axios to v1.12.0 [SECURITY] (#303)
- Update Node.js to v22.19.0 (#298)
- Update dependency next to v15.5.2 (#297)
- Update dependency eslint-config-next to v15.5.2 (#296)
- Update codecov/codecov-action action to v5.5.1 (#295)
- Update nginx Docker tag to v1.29.1 (#294)
- Update dependency starlette to v0.47.3 (#293)
- Update dependency ruff to v0.12.12 (#292)
- Update dependency Django to v4.2.24 (#291)

Version 0.14.0 (Released September 08, 2025)
--------------

- Update langgraph (#287)
- No  dupe citations (#286)
- Update dependency django-debug-toolbar to v6 (#265)
- Update dependency next to v15.4.7 [SECURITY] (#284)
- Update dependency uvicorn to ^0.35.0 (#264)

Version 0.13.2 (Released August 29, 2025)
--------------

- Better citations #3 (#281)
- fix object_id (#280)

Version 0.13.1 (Released August 28, 2025)
--------------

- Better citations (#278)
- Add canvas_token details to README (#275)

Version 0.13.0 (Released August 28, 2025)
--------------

- Display citations in syllabus bot responses (#276)

Version 0.12.1 (Released August 18, 2025)
--------------

- update open-learning-ai-tutor (#273)
- canvas tutor authentication (#272)

Version 0.12.0 (Released August 11, 2025)
--------------

- update open-learning-ai-tutor (#269)
- Add gpt-5, 5-mini to the list of models that do not like the temperature variable (#268)

Version 0.11.3 (Released August 07, 2025)
--------------

- Track LLM costs in Posthog (#254)

Version 0.11.2 (Released August 05, 2025)
--------------

- Canvas Tutorbot
- fix(deps): update dependency ruff to v0.12.7 (#263)
- fix(deps): update dependency next to v15.4.5 (#262)
- chore(deps): update node.js to v22.18.0 (#261)
- chore(deps): update dependency eslint-config-next to v15.4.5 (#260)
- fix(deps): update react monorepo to v19.1.1 (#259)
- fix(deps): update dependency django-oauth-toolkit to v3 (#14)
- chore(deps): update actions/setup-python digest to a26af69 (#159)
- chore(deps): update dependency eslint-plugin-jest to v29 (#248)
- chore(deps): update dependency jest-extended to v6 (#249)
- fix(deps): update dependency starlette to v0.47.2 [security] (#253)

Version 0.11.1 (Released July 21, 2025)
--------------

- add edx_module_id to tutorbot output (#251)
- truncate conversation (#237)

Version 0.11.0 (Released July 15, 2025)
--------------

- New Canvas-specific syllabus bot endpoint (#238)
- fix(deps): update dependency uvicorn to ^0.35.0 (#247)
- fix(deps): update dependency starlette to v0.47.1 (#246)
- fix(deps): update dependency ruff to v0.12.3 (#245)
- chore(deps): update node.js to v22.17.0 (#244)
- chore(deps): update nginx docker tag to v1.29.0 (#243)
- fix(deps): update dependency next to v15.3.5 (#242)
- fix(deps): update dependency langmem to ^0.0.28 (#241)
- chore(deps): update redis docker tag to v7.4.5 (#240)
- chore(deps): update dependency eslint-config-next to v15.3.5 (#239)
- update open-learning-ai-tutor (#236)
- fix(deps): update django-health-check digest to 5267d8f (#225)
- fix(deps): update python docker tag to v3.13.5 (#231)
- Remove pytz from unit test (#58)

Version 0.10.2 (Released July 09, 2025)
--------------

- change rc urls (#234)
- chore(deps): update dependency next to v15.3.3 [security] (#233)
- Evaluate different prompts (#232)
- RAG evaluation mgmt command (#223)
- fix(deps): update dependency next to v15.3.4 (#230)
- fix(deps): update dependency django to v4.2.23 (#229)
- chore(deps): update redis docker tag to v7.4.4 (#228)
- chore(deps): update dependency eslint-config-next to v15.3.4 (#227)
- chore(deps): update codecov/codecov-action action to v5.4.3 (#226)
- Limit message length (#224)

Version 0.10.1 (Released June 24, 2025)
--------------

- Stream tutor messages (#220)
- chore(deps): update dependency pytest-asyncio to v1 (#202)
- fix(deps): update dependency django-anymail to v13 (#139)

Version 0.10.0 (Released June 18, 2025)
--------------

- AI system prompts endpoint (#218)
- 1-word change to video prompt (#215)
- fix(deps): update dependency requests to v2.32.4 [security] (#219)
- fix(deps): update dependency django to v4.2.22 [security] (#217)
- Add architecture overview to readme (#214)

Version 0.9.3 (Released June 09, 2025)
-------------

- fix(deps): update dependency django-guardian to v3 (#203)
- Change default env values for bot models, search url (#210)

Version 0.9.2 (Released June 05, 2025)
-------------

- Revert overwrite of search_content_files change (#212)
- Syllabus bot for programs (#206)
- More summary prompt tweaking (#208)
- Use learn auth key for requests (#192)
- Enable the recommendation bot to search for specific resource details (#205)
- Tweak the syllabus/video_gpt system prompts, to avoid LLM confusion over the resource in question. (#204)
- fix(deps): update dependency ruff to v0.11.11 (#201)
- fix(deps): update dependency langmem to ^0.0.27 (#200)

Version 0.8.0 (Released May 28, 2025)
-------------

- Summarize chat sessions beyond a certain token limit (#193)

Version 0.7.0 (Released May 21, 2025)
-------------

- Adjust chatbot system prompts to tell the LLM its name is Tim (#196)
- Add langsmith integration to the README (#195)
- Rename the imported realm file name. (#197)
- config: Bypass SSL redirect for healthcheck endpoints

Version 0.6.4 (Released May 21, 2025)
-------------

- feat: Add healthcheck plugin (#188)

Version 0.6.3 (Released May 14, 2025)
-------------

- Update tutor version (#189)
- Langsmith tracing and logging (#169)
- Update dependency ruff to v0.11.9 (#185)
- Update dependency open-learning-ai-tutor to ^0.0.9 (#184)
- Update dependency next to v15.3.2 (#183)
- Update dependency eslint-config-next to v15.3.2 (#182)

Version 0.6.2 (Released May 13, 2025)
-------------

- allow newlines in display math replacement (#180)
- Update dependency Django to v4.2.21 [SECURITY] (#179)
- Start new thread when changing model in sandbox (#175)

Version 0.6.1 (Released May 08, 2025)
-------------

- use mathjax for tutor (#168)
- Make sure any exceptions are  logged (#173)

Version 0.6.0 (Released May 05, 2025)
-------------

- Fix ci vars (#176)
- Frontend related changes for new ui sandbox option (#172)
- add metadata and error display (#171)
- Option to override the default recommendation bot search url  (#157)
- Update Node.js to v22.15.0 (#163)
- Update nginx Docker tag to v1.28.0 (#162)
- Update redis Docker tag to v7.4.3 (#161)
- Update dependency ruff to v0.11.7 (#160)
- add ci env vars (#158)
- Clear throttle cache on ConsumerThrottleLimit.save (#150)

Version 0.5.5 (Released April 29, 2025)
-------------

- include credentials in edx requests (#155)
- turn off default gemini reviews (#153)
- fix two trailing slash issues (#152)
- fix chat ids
- better names
- add action env vars
- fix reset
- remove old tutor ui
- remove some logs, add a comment
- add workflow env vars
- prevent using chat while loading
- simplify resource selection, add login
- add video tab
- add assessment tab
- start tying to url
- add syllabus gpt panel
- add recommendation tab
- add openedx queries
- allow proxying local dev requests to deployed openedx
- add models api call
- add tabs with placeholder content
- use litellm for tutor
- chore(deps): lock file maintenance (#149)
- fix(deps): update react monorepo to v19.1.0 (#147)
- fix(deps): update dependency next to v15.3.1 (#146)
- fix(deps): update dependency ruff to v0.11.6 (#145)
- chore(deps): update nginx docker tag to v1.27.5 (#144)
- chore(deps): update dependency eslint-config-next to v15.3.1 (#143)

Version 0.5.2 (Released April 17, 2025)
-------------

- remove tutor problem view
- Update to open_learning_ai_tutor 0.0.6
- chore(deps): lock file maintenance (#134)
- chore(deps): update dependency pdbpp to ^0.11.0 (#138)
- fix(deps): update python docker tag to v3.13.3
- chore(deps): update codecov/codecov-action action to v5.4.2
- chore(deps): update actions/setup-python digest to 8d9ed9a
- fix(deps): update material-ui monorepo to v7
- chore(deps): update dependency eslint-import-resolver-typescript to v4
- chore(deps): update dependency pytest-asyncio to ^0.26.0 (#131)
- chore(deps): update dependency eslint-config-next to v15.3.0
- fix(deps): update dependency starlette to v0.46.2
- chore(deps): update dependency django-debug-toolbar to v5
- chore(deps): update dependency eslint-config-prettier to v10
- fix(deps): update dependency ipython to v9
- fix(deps): update dependency ruff to v0.11.5
- chore(deps): update react monorepo
- chore(deps): update actions/setup-node digest to 49933ea
- chore(deps): update dependency faker to v37

Version 0.5.1 (Released April 15, 2025)
-------------

- Chat request limits for users (#122)
- LLM models API endpoint for playground (#124)
- Update open_learning_ai_tutor

Version 0.5.0 (Released April 07, 2025)
-------------

- Change recommendation bot course links to a resource drawer link (#120)

Version 0.4.0 (Released April 03, 2025)
-------------

- fix(deps): update dependency next to v15.2.4 [security] (#117)
- Send posthog event with metadata for tutorbot (#116)
- Do not update sessions assigned to another user (#115)

Version 0.3.2 (Released March 27, 2025)
-------------

- use edx module ids to find the problem
- Update main/settings.py
- Add OTEL insecure flag
- fix(deps): update dependency next to v15.2.3 [security] (#110)

Version 0.3.1 (Released March 26, 2025)
-------------

- adding debug toolbar urls
- Fixing typo in API_BASE_URL
- Updated poetry.lock file
- standardizing config filename
- fixing compose for litellm
- fix(deps): update dependency ruff to v0.11.0 (#104)
- fix(deps): update dependency next to v15.2.2 (#103)
- chore(deps): update dependency eslint-config-next to v15.2.2 (#102)
- Add OpenTelemetry Config
- fix(deps): update python docker tag to v3.13.2 (#10)

Version 0.3.0 (Released March 12, 2025)
-------------

- fix(deps): update dependency starlette to v0.46.1 (#98)
- fix(deps): update dependency ruff to v0.9.10 (#97)
- fix(deps): update dependency next to v15.2.1 (#96)
- fix(deps): update dependency axios to v1.8.2 [security] (#95)
- chore(deps): update dependency eslint-config-next to v15.2.1 (#94)
- refactor:! edx_block_id to edx_module_id and better setting name (#91)
- Make sentry work for asgi endpoints too (#89)
- fix(deps): update dependency django to v4.2.20 [security] (#90)
- Better cookie management (#84)
- run collectstatic within the docker build
- put the static file in /static instead, and make sure the directory exists
- should go to staticfiles instead
- generate the git hash file based on a build arg and put it in /src/static/hash.txt
- feat: Add Video GPT (#56)

Version 0.2.1 (Released March 06, 2025)
-------------

- Add tutor bot frontend
- Update dependency starlette to v0.46.0 (#79)
- Update dependency next to v15.2.0 (#78)
- Update dependency langgraph to ^0.3.0 (#77)
- Update dependency eslint-config-next to v15.2.0 (#76)
- Update codecov/codecov-action action to v5.4.0 (#75)
- Update dependency ruff to v0.9.9 (#74)
- Tutor bot backend
- Fix chat UI height (#73)
- Update dependency faker to v36
- Update dependency @mitodl/smoot-design to v3

Version 0.2.0 (Released February 26, 2025)
-------------

- Update dependency starlette to ^0.46.0 (#66)
- Update Node.js to v22.14.0 (#65)
- Update nginx Docker tag to v1.27.4 (#64)
- Update dependency ruff to v0.9.7 (#63)
- Update dependency next to v15.1.7 (#62)
- Update dependency eslint-config-next to v15.1.7 (#61)
- Update dependency Django to v4.2.19 (#60)

Version 0.1.0 (Released February 21, 2025)
-------------

- Zero the version

