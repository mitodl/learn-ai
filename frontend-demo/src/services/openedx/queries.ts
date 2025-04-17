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
      queryFn: () => fetchCourseBlocks(opts).then((res) => res.data),
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
    enabled: enabled && Boolean(username && opts.blockUsageKey),
  })
}

export { queries as openEdxQueries, useV2Block }
