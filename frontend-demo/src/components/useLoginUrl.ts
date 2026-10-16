"use client"

import { useSyncExternalStore } from "react"
import { LOGIN_URL } from "@/constants"

const loginUrl = LOGIN_URL as string

const subscribe = () => () => {}

const loginUrlForCurrentPage = () => {
  const url = new URL(loginUrl)
  url.searchParams.set("next", window.location.href)
  return url.toString()
}

/**
 * The login URL, with ?next= set to the current page, so the user comes back
 * here after signing in. Server rendering gets the plain login URL.
 */
const useLoginUrl = (): string =>
  useSyncExternalStore(subscribe, loginUrlForCurrentPage, () => loginUrl)

export { useLoginUrl }
