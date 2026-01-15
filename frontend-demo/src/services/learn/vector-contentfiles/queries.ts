import { queryOptions } from "@tanstack/react-query"
import type { PaginatedContentFileList } from "@mitodl/mit-learn-api-axios/v1"
import axios from "axios"

type VectorContentListOptions = {
  q: string
  group_by: string
  platform: string
  group_size: number
}

const keys = {
  root: () => ["vector_contentfiles"],
  contentfilesListing: (opts: VectorContentListOptions) => [
    ...keys.root(),
    "listing",
    opts,
  ],
}

const queries = {
  listing: (opts: VectorContentListOptions) =>
    queryOptions({
      queryKey: keys.contentfilesListing(opts),
      /**
       * use axios to hit the proxy endpoint which adds a token
       */
      queryFn: () => {
        const search = new URLSearchParams()
        if (opts.q) {
          search.append("q", opts.q)
        }
        if (opts.platform) {
          search.append("platform", opts.platform)
        }
        if (opts.group_size) {
          search.append("group_size", opts.group_size.toString())
        }
        if (opts.group_by) {
          search.append("group_by", opts.group_by)
        }

        return axios
          .get(
            `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/learn-api/v0/vector_content_files_search?${search.toString()}`,
          )
          .then((res) => res.data as PaginatedContentFileList)
      },
    }),
}

export { queries }
