import { openEdxQueries } from "@/services/openedx"
import Alert from "@mui/material/Alert"
import Link from "@mui/material/Link"
import { useQuery } from "@tanstack/react-query"
import invariant from "tiny-invariant"

const LOGIN_URL = process.env.NEXT_PUBLIC_OPENEDX_LOGIN_URL
invariant(LOGIN_URL, "NEXT_PUBLIC_OPENEDX_LOGIN_URL is required")

const OpenEdxLoginAlert = () => {
  const userMe = useQuery(openEdxQueries.userMe())
  console.log({ userMe })
  return (
    userMe.isError && (
      <Alert severity="error">
        To use this feature, you must be logged in to OpenEdx. Please log in at{" "}
        <Link target="_blank" href={LOGIN_URL}>
          {LOGIN_URL}
        </Link>
      </Alert>
    )
  )
}

export default OpenEdxLoginAlert
