"use client"

import { ThemeProvider } from "@mitodl/smoot-design"
import { AppRouterCacheProvider } from "@mui/material-nextjs/v15-appRouter"
import { QueryClientProvider, QueryClient } from "@tanstack/react-query"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: false,
      staleTime: Infinity,
    },
  },
})

const Providers: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  return (
    <AppRouterCacheProvider>
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      </ThemeProvider>
    </AppRouterCacheProvider>
  )
}

export default Providers
