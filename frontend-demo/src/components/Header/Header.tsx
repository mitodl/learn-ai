"use client"

import MuiAppBar from "@mui/material/AppBar"
import Toolbar from "@mui/material/Toolbar"
import styled from "@emotion/styled"
import Typography from "@mui/material/Typography"
import Image from "next/image"
import MitLogo from "@/public/images/mit-logo-white.svg"
import Link from "next/link"
import { useQuery } from "@tanstack/react-query"
import { userQueries } from "@/services/ai"
import { ButtonLink } from "@mitodl/smoot-design"
import { LOGIN_URL } from "@/constants"

const AppBar = styled(MuiAppBar)(({ theme }) => ({
  padding: "16px 8px",
  backgroundColor: theme.custom.colors.navGray,
  boxShadow: "none",
  height: theme.custom.dimensions.headerHeight,
  ".MuiToolbar-root": {
    minHeight: "auto",
    height: "100%",
    gap: "16px",
  },
  [theme.breakpoints.down("sm")]: {
    height: theme.custom.dimensions.headerHeightSm,
    padding: "0",
  },
}))

const Header = () => {
  const me = useQuery(userQueries.me())
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          flex={1}
          variant="h3"
          sx={{ a: { textDecoration: "none", color: "inherit" } }}
        >
          <Link href="/">Learn AI Sandbox</Link>
        </Typography>
        {me.data?.anonymous ? (
          <ButtonLink variant="tertiary" href={LOGIN_URL}>
            Login
          </ButtonLink>
        ) : (
          <Typography
            sx={(theme) => ({
              color: theme.custom.colors.white,
            })}
            variant="body1"
          >
            {me.data?.username}
          </Typography>
        )}
        <Image height={32} src={MitLogo} alt="" />
      </Toolbar>
    </AppBar>
  )
}

export default Header
