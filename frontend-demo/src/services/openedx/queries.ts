import { queryOptions } from "@tanstack/react-query"
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

export { queries }
