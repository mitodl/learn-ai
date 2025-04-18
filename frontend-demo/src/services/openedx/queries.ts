import { queryOptions, useQuery } from "@tanstack/react-query"
import { fetchCourseBlocks, fetchUserMe } from "./client"

import type { CourseV2BlocksRequest } from "./client"

const keys = {
  root: () => ["openedx"],
  coursesV2Blocks: (opts: CourseV2BlocksRequest) => [
    ...keys.root(),
    "coursesV2Blocks",
    opts,
  ],
  userMe: () => [...keys.root(), "userMe"],
}

const queries = {
  coursesV2Blocks: (opts: CourseV2BlocksRequest) =>
    queryOptions({
      queryKey: keys.coursesV2Blocks(opts),
      queryFn: () => {
        return fetchCourseBlocks(opts).then((res) => res.data)
      },
    }),
  userMe: () => {
    return queryOptions({
      queryKey: keys.root(),
      queryFn: () => fetchUserMe(),
    })
  },
}

const useV2Block = (
  opts: Omit<CourseV2BlocksRequest, "username">,
  {
    enabled,
  }: {
    enabled?: boolean
  } = {},
) => {
  const userMe = useQuery(queries.userMe())
  const username = userMe.data?.username ?? ""
  return useQuery({
    ...queries.coursesV2Blocks({ ...opts, username }),
    // Note: enabled must come AFTER the Boolean(...)
    // true && undefined => undefined  => query enabled, since that's the default.
    // false && undefined => false => query disabled, as it should be
    // If enabled comes first, then the Boolean(...) won't always be checked.
    enabled: Boolean(username && opts.blockUsageKey) && enabled,
  })
}

export { queries as openEdxQueries, useV2Block }
