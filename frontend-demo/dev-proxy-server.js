const httpProxy = require("http-proxy")

const proxyTarget = process.env.NEXT_PUBLIC_OPENEDX_PROXY_TARGET
const cookieName = process.env.OPENEDX_SESSION_COOKIE_NAME
const cookieValue = process.env.OPENEDX_SESSION_COOKIE_VALUE

if (!proxyTarget) {
  console.log("NEXT_PUBLIC_OPENEDX_PROXY_TARGET is not set. Skipping proxy.")
  process.exit(0)
}

const proxy = httpProxy.createProxyServer({
  changeOrigin: true,
  target: proxyTarget,
  headers: {
    Cookie: `${cookieName}=${cookieValue};`,
  },
})

proxy.on("proxyReq", (proxyReq) => {
  proxyReq.removeHeader("x-forwarded-host")
  proxyReq.removeHeader("x-forwarded-proto")
  proxyReq.removeHeader("x-forwarded-port")
  proxyReq.removeHeader("x-forwarded-for")
})

proxy.listen(8004)
