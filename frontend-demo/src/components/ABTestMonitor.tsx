import React, { useEffect, useState } from "react"
import { useAiChat } from "@mitodl/smoot-design/ai"
import { Box, Portal } from "@mui/material"
import ABTestResponseSelector, { type ABTestResponse } from "./ABTestResponseSelector"
import { extractABTestData, submitABTestChoice } from "@/services/abTestChoice"

interface ABTestMonitorProps {
  problemSetTitle: string
  runReadableId: string
}

const ABTestMonitor: React.FC<ABTestMonitorProps> = ({
  problemSetTitle,
  runReadableId,
}) => {
  const { messages } = useAiChat()
  const [pendingABTest, setPendingABTest] = useState<ABTestResponse | null>(null)
  const [processedMessageIds, setProcessedMessageIds] = useState<Set<string>>(new Set())

  useEffect(() => {
    // Check the latest assistant message for A/B test data
    const assistantMessages = messages.filter(msg => msg.role === "assistant")
    const latestAssistantMessage = assistantMessages[assistantMessages.length - 1]

    if (latestAssistantMessage && latestAssistantMessage.content) {
      // Create a unique ID for this message to avoid reprocessing
      const messageId = `${latestAssistantMessage.content.slice(0, 50)}-${assistantMessages.length}`
      
      if (!processedMessageIds.has(messageId)) {
        const abTestData = extractABTestData(latestAssistantMessage.content)
        
        if (abTestData) {
          setPendingABTest(abTestData)
          setProcessedMessageIds(prev => new Set([...prev, messageId]))
        }
      }
    }
  }, [messages, processedMessageIds])

  const handleABTestChoice = async (
    chosenVariant: "control" | "treatment", 
    abTestData: ABTestResponse
  ) => {
    try {
      await submitABTestChoice({
        thread_id: abTestData.metadata.thread_id,
        chosen_variant: chosenVariant,
        ab_response_data: abTestData,
        problem_set_title: problemSetTitle,
        run_readable_id: runReadableId,
      })

      // Clear the pending A/B test
      setPendingABTest(null)
      
      console.log("A/B test choice submitted successfully", {
        variant: chosenVariant,
        testName: abTestData.metadata.test_name,
      })
    } catch (error) {
      console.error("Failed to submit A/B test choice:", error)
      throw error
    }
  }

  if (!pendingABTest) {
    return null
  }

  return (
    <Portal>
      <Box
        sx={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          zIndex: 9999,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 2,
        }}
      >
        <Box
          sx={{
            backgroundColor: "white",
            borderRadius: 2,
            padding: 3,
            maxWidth: "90vw",
            maxHeight: "90vh",
            overflow: "auto",
            boxShadow: 24,
          }}
        >
          <ABTestResponseSelector
            abTestData={pendingABTest}
            onSelect={handleABTestChoice}
          />
        </Box>
      </Box>
    </Portal>
  )
}

export default ABTestMonitor
