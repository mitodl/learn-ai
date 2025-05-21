# learn-ai

![CI Workflow](https://github.com/mitodl/learn-ai/actions/workflows/ci.yml/badge.svg)

This application provides backend API endpoints to access various AI chatbots.

**SECTIONS**

1. [Initial Setup](#initial-setup)
2. [Configuration](#configuration)
3. [Committing & Formatting](#committing--formatting)
4. [Sample Requests](#sample-requests)
5. [Langsmith Integration](#langsmith-integration)

## Initial Setup

Learn-AI follows the same [initial setup steps outlined in the common OL web app guide](https://mitodl.github.io/handbook/how-to/common-web-app-guide.html).
Run through those steps **including the addition of `/etc/hosts` aliases and the optional step for running the
`createsuperuser` command**.

- The backend app runs locally on port 8005.
- A simple frontend sandbox runs at port 8003.

You can start it by running `docker compose up`

## Configuration

Configuration can be put in the following file which is gitignored:

```
mit-learn/
  ├── env/
      ├── backend.local.env
      └── frontend.local.env
```

You will need at minimum the following environment variable to run locally:

```
# In backend.local.env
OPENAI_API_KEY=<your_openai_api_key>
```

### Frontend Configuration

Some parts of the frontend sandbox are query OpenEdx APIs. In lieue of a localally running OpenEdx instance in-sync with a Learn instance, you can proxy OpenEdx requests to an RC instance. For this to work, you must add `OPENEDX_SESSION_COOKIE_VALUE` to your `frontend.local.env` file. See `frontend.env` for details.

## Committing & Formatting

To ensure commits to GitHub are safe, first install [pre-commit](https://pre-commit.com/):

```
pip install pre_commit
pre-commit install
```

Running pre-commit can confirm your commit is safe to be pushed to GitHub and correctly formatted:

```
pre-commit run --all-files
```

To automatically install precommit hooks when cloning a repo, you can run this:

```
git config --global init.templateDir ~/.git-template
pre-commit init-templatedir ~/.git-template
```

## Sample Requests

Run the following curl command to test the HTTP recommendation agent API:

```
curl 'http://ai.open.odl.local:8002/http/recommendation_agent/' \
  -H 'Accept: */*' \
  -H 'Connection: keep-alive' \
  -H 'Origin: http://ai.open.odl.local:8002' \
  -H 'Referer: http://ai.open.odl.local:8002/' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'content-type: application/json' \
  --data-raw '{"message":"I am curious about AI applications for business"}' \
  --verbose
```

## Langsmith Integration

[Langsmith](https://smith.langchain.com) can be used to manage the AI agent prompts and monitor their usage, costs, and outputs.
To enable this functionality, you need to perform the following steps:

1. Create a free Langsmith account and obtain an API key.
2. Add the following to your backend environment variables:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=<your_api_key>
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_PROJECT=<any_project_name_you_want>
   ```

On the langsmith UI, there is a "Prompts" menu button on the left side that will load a list of
them. Each environment will have its own set of prompts (each prompt will have a "\_dev/rc/prod"
suffix). If you click on one for details, there will be an "Edit in Playground" button at top
right that will let you make/test changes. The prompts are cached in redis so if changes are
made and you want them to take effect right away, you can run a new `clear_prompt_cache` management
command.

If you need to update a prompt, you have 2 options:

- Update it directly from the LangSmith prompt UI
- Use the "update_prompt" management command (ex: `./manage.py update_prompt --prompt syllabus`). If the
  prompt already exists in LangSmith and has a different value, you will need to manually confirm
  the change. This may happen if someone had editied the prompt in the Langsmith UI, in which
  case you should consult with the editor, merge the changes together into the hardcoded `prompts.py`
  value, and then run the management command again.
