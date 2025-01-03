"use client";

import { ThemeProvider } from "@mitodl/smoot-design";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v15-appRouter";


const Providers: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  return (
      <AppRouterCacheProvider>
        <ThemeProvider>{children}</ThemeProvider>
      </AppRouterCacheProvider>
  );
};

export default Providers;
