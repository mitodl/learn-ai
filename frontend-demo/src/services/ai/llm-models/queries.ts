import { queryOptions } from "@tanstack/react-query"
import { getAPISettings } from "../settings"
import { LlmModelsApi } from "@api/v0"

const llmModelsApi = new LlmModelsApi(...getAPISettings())

const keys = {
  root: () => ["llm-models"],
  list: () => [...keys.root(), "list"],
}

const queries = {
  list: () =>
    queryOptions({
      queryKey: keys.list(),
      queryFn: () => llmModelsApi.llmModelsList().then((res) => res.data),
    }),
}

export { queries }
