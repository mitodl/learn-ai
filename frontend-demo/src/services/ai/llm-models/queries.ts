import { queryOptions } from "@tanstack/react-query"
import { llmModelsApi } from "../clients"

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
