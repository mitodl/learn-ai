"use client"

import type { MouseEvent } from "react"
import { LOGIN_URL } from "@/constants"

const loginUrl = LOGIN_URL as string

/**
 * Props for a Login link that returns the user to the page they're on.
 *
 * ``next`` is set when the link is clicked, not when it renders, so it follows
 * URL changes the page makes without re-rendering (tab selection uses
 * history.pushState) and works before hydration falls back to plain login.
 */
const useLoginLinkProps = () => ({
  href: loginUrl,
  onClick: (event: MouseEvent<HTMLAnchorElement>) => {
    const url = new URL(loginUrl, window.location.href)
    url.searchParams.set("next", window.location.href)
    event.currentTarget.href = url.toString()
  },
})

export { useLoginLinkProps }
