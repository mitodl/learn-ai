"use client"
import { TabContext, TabList, TabPanel } from "@mui/lab"
import type { TabListProps } from "@mui/lab/TabList"
import Tab from "@mui/material/Tab"
import { Box } from "@mui/material"
import { useSearchParams } from "next/navigation"

export enum ChatTab {
  RecommendationGPT = "RecommendationGPT",
  SyllabusGPT = "SyllabusGPT",
  VideoGPT = "VideoGPT",
  AssessmentGPT = "AssessmentGPT",
}

const ChatTabs = () => {
  const searchParams = useSearchParams()
  const currentTab = searchParams.get("tab") || ChatTab.RecommendationGPT

  const handleTabChange: TabListProps["onChange"] = (_event, newValue) => {
    const url = new URL(window.location.href)
    url.searchParams.set("tab", newValue)
    window.history.pushState({}, "", url.toString())
  }

  return (
    <TabContext value={currentTab}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <TabList onChange={handleTabChange} aria-label="Home Page Tabs">
          <Tab label="RecommendationGPT" value={ChatTab.RecommendationGPT} />
          <Tab label="SyllabusGPT" value={ChatTab.SyllabusGPT} />
          <Tab label="VideoGPT" value={ChatTab.VideoGPT} />
          <Tab label="AssessmentGPT" value={ChatTab.AssessmentGPT} />
        </TabList>
      </Box>
      <TabPanel value={ChatTab.RecommendationGPT}>
        RecommendationGPT Content
      </TabPanel>
      <TabPanel value={ChatTab.SyllabusGPT}>SyllabusGPT Content</TabPanel>
      <TabPanel value={ChatTab.VideoGPT}>VideoGPT Content</TabPanel>
      <TabPanel value={ChatTab.AssessmentGPT}>AssessmentGPT Content</TabPanel>
    </TabContext>
  )
}

export default ChatTabs
