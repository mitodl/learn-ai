import axios from "axios"
import type { AxiosResponse } from "axios"

type CourseV2BlocksRequest = {
  blockUsageKey: string
}
type CourseV2BlocksResponse = {
  root: string
  blocks: {
    [blockUsageKey: string]: {
      id: string // aka blockUsageKey
      block_id: string
      lms_web_url: string
      legacy_web_url: string
      student_view_url: string
      type: string
      display_name: string
    }
  }
}

const fetchCourseBlocks = ({
  blockUsageKey,
}: CourseV2BlocksRequest): Promise<AxiosResponse<CourseV2BlocksResponse>> => {
  const baseUrl = process.env.NEXT_PUBLIC_OPENEDX_API_BASE_URL
  if (!baseUrl) {
    throw new Error("NEXT_PUBLIC_OPENEDX_API_BASE_URL is not defined")
  }

  const url = `${baseUrl}api/courses/v2/blocks/${blockUsageKey}?depth=all&all_blocks=true`
  return axios.get<CourseV2BlocksResponse>(url)
}

export { fetchCourseBlocks }
export type { CourseV2BlocksRequest, CourseV2BlocksResponse }
