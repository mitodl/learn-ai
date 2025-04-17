import { queryOptions } from "@tanstack/react-query"
import { axiosInstance, AI_API_BASE_URL } from "../settings"

const keys = {
  root: () => ["user"],
  me: () => [...keys.root(), "me"],
}

type UserResponse = {
  username?: string
  anonymous?: boolean
}

const queries = {
  me: () =>
    queryOptions({
      queryKey: keys.me(),
      queryFn: (): Promise<UserResponse> => {
        return axiosInstance
          .get(`${AI_API_BASE_URL}/me`)
          .then((res) => res.data)
      },
    }),
}

export { queries }
