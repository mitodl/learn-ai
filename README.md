# learn-ai

![CI Workflow](https://github.com/mitodl/learn-ai/actions/workflows/ci.yml/badge.svg)

This application provides backend API endpoints to access various AI chatbots.

**SECTIONS**

1. [Initial Setup](#initial-setup)
2. [Configuration](#configuration)
3. [Committing & Formatting](#committing--formatting)
4. [Sample Requests](#sample-requests)

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
