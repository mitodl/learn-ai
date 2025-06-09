import { queryOptions } from "@tanstack/react-query"
import { getAPISettings } from "../settings"
import { PromptsApi } from "@api/v0"

const promptsApi = new PromptsApi(...getAPISettings())

const keys = {
  root: () => ["prompts"],
  list: () => [...keys.root(), "list"],
  get: (promptName: string) => [...keys.root(), "get", promptName],
}

const queries = {
  list: () =>
    queryOptions({
      queryKey: keys.list(),
      queryFn: () => promptsApi.promptsList().then((res) => res.data),
    }),
  get: (promptName: string) =>
    queryOptions({
      queryKey: keys.get(promptName),
      queryFn: () =>
        promptsApi
          .promptsRetrieve({ prompt_name: promptName })
          .then((res) => res.data),
    }),
}

export { queries }
