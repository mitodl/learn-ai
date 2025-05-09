"use client"
import { TabContext, TabList, TabPanel } from "@mui/lab"
import type { TabListProps } from "@mui/lab/TabList"
import Tab from "@mui/material/Tab"
import { Box } from "@mui/material"
import { useSearchParams } from "next/navigation"
import RecommendationContent from "./RecommendationContent"
import SyllabusContent from "./SyllabusContent"
import AssessmentContent from "./AssessmentContent"
import VideoContent from "./VideoContent"
import { useEffect, useState } from "react"

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

  const [hasRendered, setHasRendered] = useState(false)
  useEffect(() => {
    setHasRendered(true)
  }, [])
  if (!hasRendered) {
    return null
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
        <RecommendationContent />
      </TabPanel>
      <TabPanel value={ChatTab.SyllabusGPT}>
        <SyllabusContent />
      </TabPanel>
      <TabPanel value={ChatTab.VideoGPT}>
        <VideoContent />
      </TabPanel>
      <TabPanel value={ChatTab.AssessmentGPT}>
        <AssessmentContent />
      </TabPanel>
    </TabContext>
  )
}

export default ChatTabs
