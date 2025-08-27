import React, { useState } from "react"
import { Box, Button, Card, CardContent, Typography, Grid2 as Grid, Alert } from "@mui/material"
import { styled } from "@mitodl/smoot-design"

interface ABTestResponse {
  type: "ab_test_response"
  control: {
    content: string
    variant: "control"
  }
  treatment: {
    content: string
    variant: "treatment"
  }
  metadata: {
    test_name: string
    thread_id: string
    original_message: string
    edx_module_id?: string
    problem_set_title: string
    run_readable_id: string
  }
  _control_history: unknown[]
  _treatment_history: unknown[]
  _intent_history: unknown[]
  _assessment_history: unknown[]
}

interface ABTestResponseSelectorProps {
  abTestData: ABTestResponse
  onSelect: (chosenVariant: "control" | "treatment", responseData: ABTestResponse) => Promise<void>
}

const ResponseCard = styled(Card)(({ theme }) => ({
  height: "100%",
  border: `2px solid ${theme.palette.grey[300]}`,
  "&:hover": {
    borderColor: theme.palette.primary.main,
    boxShadow: theme.shadows[4],
  },
  "&.selected": {
    borderColor: theme.palette.primary.main,
    backgroundColor: `${theme.palette.primary.light}10`,
  },
}))

const ChoiceButton = styled(Button)(({ theme }) => ({
  marginTop: theme.spacing(2),
  width: "100%",
  ...theme.typography.button,
}))

const ABTestResponseSelector: React.FC<ABTestResponseSelectorProps> = ({
  abTestData,
  onSelect,
}) => {
  const [selectedVariant, setSelectedVariant] = useState<"control" | "treatment" | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [hasSubmitted, setHasSubmitted] = useState(false)

  const handleSelect = async (variant: "control" | "treatment") => {
    if (hasSubmitted) return

    setIsSubmitting(true)
    try {
      await onSelect(variant, abTestData)
      setSelectedVariant(variant)
      setHasSubmitted(true)
    } catch (error) {
      console.error("Failed to submit choice:", error)
      // Could add error state here
    } finally {
      setIsSubmitting(false)
    }
  }

  if (hasSubmitted) {
    // Show the selected response
    const chosenResponse = selectedVariant === "control" ? abTestData.control : abTestData.treatment
    return (
      <Box sx={{ my: 2 }}>
        <Alert severity="success" sx={{ mb: 2 }}>
          You selected Response {selectedVariant === "control" ? "A" : "B"}. The conversation will continue with your chosen response.
        </Alert>
        <Card sx={{ border: 2, borderColor: "primary.main" }}>
          <CardContent>
            <Typography variant="body1" component="div">
              {chosenResponse?.content}
            </Typography>
          </CardContent>
        </Card>
      </Box>
    )
  }

  return (
    <Box sx={{ my: 2 }}>
      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="body2">
          <strong>Choose your preferred response:</strong> Please select which response (A or B) was more helpful to you. 
          The conversation will continue with your chosen response.
        </Typography>
      </Alert>
      
      <Grid container spacing={2}>
        <Grid size={6}>
          <ResponseCard className={selectedVariant === "control" ? "selected" : ""}>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2, color: "primary.main" }}>
                Response A
              </Typography>
              <Typography variant="body1" component="div" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                {abTestData.control.content}
              </Typography>
              <ChoiceButton
                variant={selectedVariant === "control" ? "contained" : "outlined"}
                color="primary"
                onClick={() => handleSelect("control")}
                disabled={isSubmitting}
              >
                {isSubmitting && selectedVariant === "control" ? "Selecting..." : "Choose Response A"}
              </ChoiceButton>
            </CardContent>
          </ResponseCard>
        </Grid>
        
        <Grid size={6}>
          <ResponseCard className={selectedVariant === "treatment" ? "selected" : ""}>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2, color: "primary.main" }}>
                Response B
              </Typography>
              <Typography variant="body1" component="div" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                {abTestData.treatment.content}
              </Typography>
              <ChoiceButton
                variant={selectedVariant === "treatment" ? "contained" : "outlined"}
                color="primary"
                onClick={() => handleSelect("treatment")}
                disabled={isSubmitting}
              >
                {isSubmitting && selectedVariant === "treatment" ? "Selecting..." : "Choose Response B"}
              </ChoiceButton>
            </CardContent>
          </ResponseCard>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ABTestResponseSelector
export type { ABTestResponse }
