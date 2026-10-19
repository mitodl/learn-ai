"use client"

import { LOGIN_URL } from "@/constants"

const loginUrl = LOGIN_URL as string

/**
 * Props for a Login link that returns the user to the page they're on.
 *
 * ``next`` is set when the link is pressed, not when it renders, so it follows
 * URL changes the page makes without re-rendering (tab selection uses
 * history.pushState) and works before hydration falls back to plain login.
 * Middle-click and "Open link in new tab" skip onClick, so pointer down and
 * the context menu (including the keyboard's) set it too.
 */
const setNext = ({ currentTarget }: { currentTarget: HTMLAnchorElement }) => {
  const url = new URL(loginUrl, window.location.href)
  url.searchParams.set("next", window.location.href)
  currentTarget.href = url.toString()
}

const loginLinkProps = {
  href: loginUrl,
  onPointerDown: setNext,
  onContextMenu: setNext,
  onClick: setNext,
}

export { loginLinkProps }
