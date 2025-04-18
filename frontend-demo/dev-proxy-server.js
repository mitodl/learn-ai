const httpProxy = require("http-proxy")

const proxyTarget = process.env.NEXT_PUBLIC_OPENEDX_PROXY_TARGET
const cookieName = process.env.OPENEDX_SESSION_COOKIE_NAME
const cookieValue = process.env.OPENEDX_SESSION_COOKIE_VALUE

if (!proxyTarget) {
  console.log("NEXT_PUBLIC_OPENEDX_PROXY_TARGET is not set. Skipping proxy.")
  process.exit(0)
}

httpProxy
  .createProxyServer({
    changeOrigin: true,
    target: proxyTarget,
    headers: {
      Cookie: `${cookieName}=${cookieValue};`,
    },
  })
  .listen(8004)
