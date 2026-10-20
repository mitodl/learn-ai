import { fireEvent, render, screen } from "@testing-library/react"
import { loginLinkProps } from "./loginLinkProps"

jest.mock("@/constants", () => ({ LOGIN_URL: "https://api.test/http/login/" }))

const expectedHref = () =>
  `https://api.test/http/login/?next=${encodeURIComponent(window.location.href)}`

test.each([
  ["pointer down", fireEvent.pointerDown],
  ["context menu", fireEvent.contextMenu],
])("%s sets next to the current URL", (_name, fire) => {
  render(<a {...loginLinkProps}>Login</a>)
  const link = screen.getByRole("link")
  expect(link).toHaveAttribute("href", "https://api.test/http/login/")

  window.history.pushState({}, "", "/chat?tab=syllabus")
  fire(link)

  expect(link).toHaveAttribute("href", expectedHref())
})
