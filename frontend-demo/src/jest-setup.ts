import * as matchers from "jest-extended"
import failOnConsole from "jest-fail-on-console"
import { resetAllWhenMocks } from "jest-when"
import "@testing-library/jest-dom"

expect.extend(matchers)
failOnConsole()

afterEach(() => {
  /**
   * Clear all mock call counts between tests.
   * This does NOT clear mock implementations.
   * Mock implementations are always cleared between test files.
   */
  jest.clearAllMocks()
  resetAllWhenMocks()
  window.history.replaceState({}, "", "/")
})
