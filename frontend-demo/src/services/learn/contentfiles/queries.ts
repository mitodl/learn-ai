import { queryOptions } from "@tanstack/react-query"
import type { PaginatedContentFileList } from "@mitodl/mit-learn-api-axios/v1"
import axios from "axios"

type ContentListOptions = {
  edxModuleIds: string[]
}

const keys = {
  root: () => ["contentfiles"],
  contentfilesListing: (opts: ContentListOptions) => [
    ...keys.root(),
    "listing",
    opts,
  ],
}

const queries = {
  listing: (opts: ContentListOptions) =>
    queryOptions({
      queryKey: keys.contentfilesListing(opts),
      /**
       * TODO:
       * There's a bug in @mitodl/mit-learn-api-axios... it thinks /contentfiles/ listing
       * API requires a path parameter.
       *
       * For now, use axios.get instead.
       */
      queryFn: () => {
        const search = new URLSearchParams()
        opts.edxModuleIds.forEach((id) => {
          search.append("edx_module_id", id)
        })
        return axios
          .get(
            `${process.env.NEXT_PUBLIC_MIT_LEARN_API_BASE_URL}/api/v1/contentfiles?${search.toString()}`,
          )
          .then((res) => res.data as PaginatedContentFileList)
      },
    }),
}

export { queries }
