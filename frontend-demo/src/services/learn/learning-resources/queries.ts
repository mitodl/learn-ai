import { queryOptions } from "@tanstack/react-query"
import { getAPISettings } from "../settings"
import { LearningResourcesApi } from "@mitodl/open-api-axios/v1"
import type { LearningResourcesApiLearningResourcesRetrieveRequest } from "@mitodl/open-api-axios/v1"

const learningResources = new LearningResourcesApi(...getAPISettings())

const keys = {
  root: () => ["learning_resources"],
  retrieve: (opts: LearningResourcesApiLearningResourcesRetrieveRequest) => [
    ...keys.root(),
    "retrieve",
    opts,
  ],
}

const queries = {
  retrieve: (opts: LearningResourcesApiLearningResourcesRetrieveRequest) =>
    queryOptions({
      queryKey: keys.retrieve(opts),
      queryFn: () =>
        learningResources.learningResourcesRetrieve(opts).then((r) => r.data),
    }),
}

export { queries }
