import type { ABTestResponse } from "@/components/ABTestResponseSelector"

interface ABTestChoiceRequest {
  thread_id: string
  chosen_variant: "control" | "treatment"
  ab_response_data: ABTestResponse
  user_preference_reason?: string
  problem_set_title: string
  run_readable_id: string
}

interface ABTestChoiceResponse {
  success: boolean
  message: string
  chosen_variant: "control" | "treatment"
  chosen_content: string
}

export const submitABTestChoice = async (
  choiceData: ABTestChoiceRequest
): Promise<ABTestChoiceResponse> => {
  const baseUrl = process.env.NEXT_PUBLIC_MITOL_API_BASE_URL ?? ""
  const url = `${baseUrl}/api/v0/ab_test_choice/`

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    body: JSON.stringify(choiceData),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: "Unknown error" }))
    throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`)
  }

  return response.json()
}

export const extractABTestData = (messageContent: string): ABTestResponse | null => {
  // Look for JSON data in HTML comments
  const commentMatch = messageContent.match(/<!--\s*(\{.*?\})\s*-->/s)
  
  if (!commentMatch) {
    return null
  }

  try {
    const jsonData = JSON.parse(commentMatch[1])
    
    // Verify it's an A/B test response
    if (jsonData.type === "ab_test_response" && jsonData.control && jsonData.treatment) {
      return jsonData as ABTestResponse
    }
  } catch (error) {
    console.error("Failed to parse A/B test data:", error)
  }

  return null
}

export type { ABTestChoiceRequest, ABTestChoiceResponse }
