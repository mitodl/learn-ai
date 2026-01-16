"use client"
import { TabContext, TabList, TabPanel } from "@mui/lab"
import type { TabListProps } from "@mui/lab/TabList"
import Tab from "@mui/material/Tab"
import { Box } from "@mui/material"
import { useSearchParams } from "next/navigation"
import RecommendationContent from "./RecommendationContent"
import SyllabusContent from "./SyllabusContent"
import SyllabusCanvasContent from "./SyllabusCanvasContent"
import AssessmentContent from "./AssessmentContent"
import CanvasAssessmentContent from "./CanvasAssessmentContent"
import VideoContent from "./VideoContent"
import { useSyncExternalStore } from "react"

export enum ChatTab {
  RecommendationGPT = "RecommendationGPT",
  SyllabusGPT = "SyllabusGPT",
  SyllabusCanvasGPT = "SyllabusCanvasGPT",
  VideoGPT = "VideoGPT",
  AssessmentGPT = "AssessmentGPT",
  CanvasAssessmentGPT = "CanvasAssessmentGPT",
}

const ChatTabs = () => {
  const searchParams = useSearchParams()
  const currentTab = searchParams.get("tab") || ChatTab.RecommendationGPT

  const handleTabChange: TabListProps["onChange"] = (_event, newValue) => {
    const url = new URL(window.location.href)
    url.searchParams.set("tab", newValue)
    window.history.pushState({}, "", url.toString())
  }

  /*
  On the server (SSR):
    React calls getServerSnapshot() → returns false → component returns null

  On the client:
    React calls getSnapshot() → returns true → component renders normally

  subscribe is a no-op because there's no external store to subscribe to
  */
  const hasRendered = useSyncExternalStore(
    () => () => {}, // subscribe: no-op
    () => true, // getSnapshot: always returns true on client
    () => false, // getServerSnapshot: always returns false on server
  )
  if (!hasRendered) {
    return null
  }

  return (
    <TabContext value={currentTab}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <TabList onChange={handleTabChange} aria-label="Home Page Tabs">
          <Tab label="RecommendationGPT" value={ChatTab.RecommendationGPT} />
          <Tab label="SyllabusGPT" value={ChatTab.SyllabusGPT} />
          <Tab label="SyllabusCanvasGPT" value={ChatTab.SyllabusCanvasGPT} />
          <Tab label="VideoGPT" value={ChatTab.VideoGPT} />
          <Tab label="AssessmentGPT" value={ChatTab.AssessmentGPT} />
          <Tab
            label="CanvasAssessmentGPT"
            value={ChatTab.CanvasAssessmentGPT}
          />
        </TabList>
      </Box>
      <TabPanel value={ChatTab.RecommendationGPT}>
        <RecommendationContent />
      </TabPanel>
      <TabPanel value={ChatTab.SyllabusGPT}>
        <SyllabusContent />
      </TabPanel>
      <TabPanel value={ChatTab.SyllabusCanvasGPT}>
        <SyllabusCanvasContent />
      </TabPanel>
      <TabPanel value={ChatTab.VideoGPT}>
        <VideoContent />
      </TabPanel>
      <TabPanel value={ChatTab.AssessmentGPT}>
        <AssessmentContent />
      </TabPanel>
      <TabPanel value={ChatTab.CanvasAssessmentGPT}>
        <CanvasAssessmentContent />
      </TabPanel>
    </TabContext>
  )
}

export default ChatTabs
