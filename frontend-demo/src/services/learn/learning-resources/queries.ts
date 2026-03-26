import { queryOptions } from "@tanstack/react-query"
import { getAPISettings } from "../settings"
import {
  LearningResourcesApi,
  LearningResourcesSearchApi,
} from "@mitodl/mit-learn-api-axios/v1"
import type {
  LearningResourcesApiLearningResourcesRetrieveRequest,
  LearningResourcesSearchApiLearningResourcesSearchRetrieveRequest,
} from "@mitodl/mit-learn-api-axios/v1"

const learningResources = new LearningResourcesApi(...getAPISettings())
const learningResourcesSearch = new LearningResourcesSearchApi(
  ...getAPISettings(),
)
const keys = {
  root: () => ["learning_resources"],
  retrieve: (opts: LearningResourcesApiLearningResourcesRetrieveRequest) => [
    ...keys.root(),
    "retrieve",
    opts,
  ],
  search: (
    opts: LearningResourcesSearchApiLearningResourcesSearchRetrieveRequest,
  ) => [...keys.root(), "search", opts],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  vectorSearch: (opts: any) => [...keys.root(), "vectorSearch", opts],
}

const queries = {
  retrieve: (opts: LearningResourcesApiLearningResourcesRetrieveRequest) =>
    queryOptions({
      queryKey: keys.retrieve(opts),
      queryFn: () =>
        learningResources.learningResourcesRetrieve(opts).then((r) => r.data),
    }),
  search: (
    opts: LearningResourcesSearchApiLearningResourcesSearchRetrieveRequest,
  ) =>
    queryOptions({
      queryKey: keys.search(opts),
      queryFn: () =>
        learningResourcesSearch
          .learningResourcesSearchRetrieve(opts)
          .then((r) => r.data),
    }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  vectorSearch: (opts: any) =>
    queryOptions({
      queryKey: keys.vectorSearch(opts),
      queryFn: () => {
        const search = new URLSearchParams()
        if (opts.q) search.append("q", opts.q)
        if (opts.limit) search.append("limit", opts.limit.toString())
        if (opts.offset) search.append("offset", opts.offset.toString())
        if (opts.professional !== undefined)
          search.append("professional", opts.professional.toString())
        if (opts.free !== undefined) search.append("free", opts.free.toString())
        if (opts.certification !== undefined)
          search.append("certification", opts.certification.toString())
        if (opts.hybrid_search !== undefined)
          search.append("hybrid_search", opts.hybrid_search.toString())
        if (opts.topic)
          opts.topic.forEach((t: string) => search.append("topic", t))
        if (opts.delivery)
          opts.delivery.forEach((d: string) => search.append("delivery", d))
        if (opts.resource_type_group) {
          const types = Array.isArray(opts.resource_type_group)
            ? opts.resource_type_group
            : [opts.resource_type_group]
          types.forEach((rtg: string) =>
            search.append("resource_type_group", rtg),
          )
        }
        if (opts.certification_type)
          opts.certification_type.forEach((c: string) =>
            search.append("certification_type", c),
          )
        if (opts.offered_by)
          opts.offered_by.forEach((o: string) => search.append("offered_by", o))
        if (opts.department)
          opts.department.forEach((d: string) => search.append("department", d))

        return import("axios").then((axios) => {
          return axios.default
            .get(
              `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/learn-api/v0/vector_learning_resources_search?${search.toString()}`,
              { withCredentials: true },
            )
            .then((r) => r.data)
        })
      },
    }),
}

export { queries }
