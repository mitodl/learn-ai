import invariant from "tiny-invariant"

const LOGIN_URL = process.env.NEXT_PUBLIC_MIT_LEARN_AI_LOGIN_URL
invariant(LOGIN_URL, "NEXT_PUBLIC_MIT_LEARN_AI_LOGIN_URL is not defined")

export { LOGIN_URL }
