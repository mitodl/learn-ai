import type { Metadata } from "next"
import { Suspense } from "react"
import Header from "@/components/Header/Header"
import "./global.css"
import Providers from "./providers"
import Container from "@mui/material/Container"

export const metadata: Metadata = {
  title: "Learn AI Demo",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="stylesheet" href="https://use.typekit.net/lbk1xay.css" />
      </head>
      <body>
        <Providers>
          <Header />
          <Suspense>
            <main>
              <Container>{children}</Container>
            </main>
          </Suspense>
        </Providers>
      </body>
    </html>
  )
}
