import { queryOptions } from "@tanstack/react-query"
// Import the correct named export for the API class
import { LearningResourcesApi } from "@mitodl/mitxonline-api-axios/v1"
// If this does not work, check the package documentation for the correct export name
// Replace with the correct type import or define locally if not exported by the package
// import type { LearningResourcesApiLearningResourcesRetrieveRequest } from "@mitodl/mitxonline-api-axios/v1"
type LearningResourcesApiLearningResourcesRetrieveRequest = {
  id: number
}

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
        LearningResourcesApi.learningResourcesRetrieve(opts).then(
          (r) => r.data,
        ),
    }),
}

export { queries }
