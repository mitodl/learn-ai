Release Notes
=============

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

