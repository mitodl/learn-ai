import { queryOptions } from "@tanstack/react-query"
import { fetchCourseBlocks } from "./client"

import type { CourseV2BlocksRequest } from "./client"

const keys = {
  root: () => ["openedx"],
  coursesV2Blocks: (opts: CourseV2BlocksRequest) => [
    ...keys.root(),
    "coursesV2Blocks",
    opts,
  ],
}

const queries = {
  coursesV2Blocks: (opts: CourseV2BlocksRequest) =>
    queryOptions({
      queryKey: keys.coursesV2Blocks(opts),
      queryFn: () => fetchCourseBlocks(opts).then((res) => res.data),
    }),
}

export { queries }
