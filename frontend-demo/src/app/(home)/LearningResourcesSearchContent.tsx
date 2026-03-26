import React, { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { queries as LearningResourcesQueries } from "@/services/learn/learning-resources/queries"
import { userQueries } from "@/services/ai"
import {
  Card,
  Typography,
  CircularProgress,
  Link,
  Checkbox,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  Button,
} from "@mui/material"
import Grid from "@mui/material/Grid"
import Box from "@mui/material/Box"
import {
  RiBookOpenFill,
  RiBriefcaseFill,
  RiArrowDownSLine,
  RiListCheck,
  RiBookmarkLine,
} from "@remixicon/react"
import {
  SearchInput,
  type SearchInputProps,
} from "@/components/SearchInput/SearchInput"
import { ButtonLink } from "@mitodl/smoot-design"
import { LOGIN_URL } from "@/constants"

const LearningResourcesSearchContent: React.FC = () => {
  const me = useQuery(userQueries.me())

  // Text search
  const [inputValue, setInputValue] = useState("")
  const [searchQuery, setSearchQuery] = useState("")

  // Filters State
  const [audience, setAudience] = useState<"all" | "academic" | "professional">(
    "all",
  )
  const [isFree, setIsFree] = useState(false)
  const [topicSearch, setTopicSearch] = useState("")
  const [selectedTopics, setSelectedTopics] = useState<string[]>([])
  const [selectedCertificates, setSelectedCertificates] = useState<string[]>([])
  const [selectedFormats, setSelectedFormats] = useState<string[]>([])
  const [selectedProviders, setSelectedProviders] = useState<string[]>([])

  const [hybridSearch, setHybridSearch] = useState(false)

  // Resource Tab Types
  const [resourceTypeGroup, setResourceTypeGroup] = useState<
    "all" | "course" | "program" | "learning_material"
  >("all")

  // Fetching data
  const { data, isLoading, error } = useQuery({
    ...LearningResourcesQueries.vectorSearch({
      q: searchQuery || undefined,
      professional:
        audience === "professional"
          ? true
          : audience === "academic"
            ? false
            : undefined,
      free: isFree ? true : undefined,
      certification_type:
        selectedCertificates.length > 0 ? selectedCertificates : undefined,
      topic: selectedTopics.length > 0 ? selectedTopics : undefined,
      delivery: selectedFormats.length > 0 ? selectedFormats : undefined,
      offered_by: selectedProviders.length > 0 ? selectedProviders : undefined,
      resource_category:
        resourceTypeGroup !== "all" ? [resourceTypeGroup] : undefined,
      hybrid_search: hybridSearch || undefined,
      limit: 30,
    }),
    enabled: true,
  })

  const handleSubmit: SearchInputProps["onSubmit"] = (e) => {
    e.preventDefault()
    setSearchQuery(inputValue)
  }

  if (me.data?.anonymous) {
    return (
      <>
        <Typography
          flex={1}
          variant="h5"
          sx={{ a: { textDecoration: "none", color: "inherit" } }}
        >
          You must&nbsp;
          <ButtonLink variant="tertiary" href={LOGIN_URL}>
            Login
          </ButtonLink>
          &nbsp;to use this feature
        </Typography>
      </>
    )
  }

  const handleClearAll = () => {
    setAudience("all")
    setIsFree(false)
    setSelectedCertificates([])
    setSelectedTopics([])
    setSelectedFormats([])
    setSelectedProviders([])
    setHybridSearch(false)
  }

  const handleAudienceChange = (
    event: React.MouseEvent<HTMLElement>,
    newAudience: "all" | "academic" | "professional",
  ) => {
    if (newAudience !== null) {
      setAudience(newAudience)
    }
  }

  const renderFilterSidebar = () => (
    <Box sx={{ pr: 3, borderRight: "1px solid #e0e0e0" }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            gap: 1,
          }}
        >
          Filter <RiListCheck size={20} />
        </Typography>
        <Button
          size="small"
          variant="text"
          onClick={handleClearAll}
          sx={{ textTransform: "none", color: "text.secondary" }}
        >
          Clear all
        </Button>
      </Box>

      <ToggleButtonGroup
        value={audience}
        exclusive
        onChange={handleAudienceChange}
        fullWidth
        sx={{ mb: 2, bgcolor: "#f5f5f5" }}
        size="small"
      >
        <ToggleButton value="all" sx={{ textTransform: "none" }}>
          All
        </ToggleButton>
        <ToggleButton value="academic" sx={{ textTransform: "none", gap: 1 }}>
          <RiBookOpenFill size={16} /> Academic
        </ToggleButton>
        <ToggleButton
          value="professional"
          sx={{ textTransform: "none", gap: 1 }}
        >
          <RiBriefcaseFill size={16} /> Professional
        </ToggleButton>
      </ToggleButtonGroup>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Content developed from MIT's academic and professional curriculum
      </Typography>

      <Box sx={{ mb: 3 }}>
        <Card variant="outlined" sx={{ mb: 2 }}>
          <Box sx={{ px: 2, py: 1.5 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={isFree}
                  onChange={(e) => setIsFree(e.target.checked)}
                />
              }
              label={<Typography>Free</Typography>}
              sx={{ width: "100%", m: 0 }}
            />
          </Box>
        </Card>

        <Accordion defaultExpanded variant="outlined" sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<RiArrowDownSLine />}>
            <Typography fontWeight="bold">Certificate</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {[
              { label: "No Certificate", value: "none" },
              { label: "Professional Certificate", value: "professional" },
              { label: "Certificate of Completion", value: "completion" },
              { label: "MicroMasters Credential", value: "micromasters" },
            ].map((cert) => (
              <FormControlLabel
                key={cert.value}
                control={
                  <Checkbox
                    checked={selectedCertificates.includes(cert.value)}
                    onChange={(e) => {
                      if (e.target.checked)
                        setSelectedCertificates([
                          ...selectedCertificates,
                          cert.value,
                        ])
                      else
                        setSelectedCertificates(
                          selectedCertificates.filter((c) => c !== cert.value),
                        )
                    }}
                  />
                }
                label={<Typography variant="body2">{cert.label}</Typography>}
                sx={{ width: "100%", m: 0 }}
              />
            ))}
          </AccordionDetails>
        </Accordion>

        <Accordion defaultExpanded variant="outlined" sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<RiArrowDownSLine />}>
            <Typography fontWeight="bold">Topic</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TextField
              size="small"
              fullWidth
              placeholder="Search Topic"
              value={topicSearch}
              onChange={(e) => setTopicSearch(e.target.value)}
              sx={{ mb: 2 }}
            />
            {[
              "Corporate Innovation",
              "Product Innovation",
              "CyberSecurity",
              "Immunology",
              "Real Estate",
              "Geography",
              "Innovation Ecosystems",
              "Natural Climate Systems",
              "Religion",
              "Pharmacology & Toxicology",
              "Theater",
              "Inventions & Patents",
              "Family Enterprise",
            ]
              .filter((topic) =>
                topic.toLowerCase().includes(topicSearch.toLowerCase()),
              )
              .map((topic) => (
                <FormControlLabel
                  key={topic}
                  control={
                    <Checkbox
                      checked={selectedTopics.includes(topic)}
                      onChange={(e) => {
                        if (e.target.checked)
                          setSelectedTopics([...selectedTopics, topic])
                        else
                          setSelectedTopics(
                            selectedTopics.filter((t) => t !== topic),
                          )
                      }}
                    />
                  }
                  label={<Typography variant="body2">{topic}</Typography>}
                  sx={{ width: "100%", m: 0 }}
                />
              ))}
          </AccordionDetails>
        </Accordion>

        <Accordion defaultExpanded variant="outlined" sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<RiArrowDownSLine />}>
            <Typography fontWeight="bold">Format</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {[
              { label: "Online", value: "online" },
              { label: "In-Person", value: "in_person" },
              { label: "Hybrid", value: "hybrid" },
            ].map((format) => (
              <FormControlLabel
                key={format.value}
                control={
                  <Checkbox
                    checked={selectedFormats.includes(format.value)}
                    onChange={(e) => {
                      if (e.target.checked)
                        setSelectedFormats([...selectedFormats, format.value])
                      else
                        setSelectedFormats(
                          selectedFormats.filter((f) => f !== format.value),
                        )
                    }}
                  />
                }
                label={<Typography variant="body2">{format.label}</Typography>}
                sx={{ width: "100%", m: 0 }}
              />
            ))}
          </AccordionDetails>
        </Accordion>

        <Accordion defaultExpanded variant="outlined" sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<RiArrowDownSLine />}>
            <Typography fontWeight="bold">Offered By</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {[
              { label: "MIT OpenCourseWare", value: "ocw" },
              { label: "MITx", value: "mitx" },
              { label: "MIT Sloan Executive Education", value: "see" },
              { label: "MIT Professional Education", value: "mitpe" },
              { label: "MIT xPRO", value: "xpro" },
            ].map((provider) => (
              <FormControlLabel
                key={provider.value}
                control={
                  <Checkbox
                    checked={selectedProviders.includes(provider.value)}
                    onChange={(e) => {
                      if (e.target.checked)
                        setSelectedProviders([
                          ...selectedProviders,
                          provider.value,
                        ])
                      else
                        setSelectedProviders(
                          selectedProviders.filter((p) => p !== provider.value),
                        )
                    }}
                  />
                }
                label={
                  <Typography variant="body2">{provider.label}</Typography>
                }
                sx={{ width: "100%", m: 0 }}
              />
            ))}
          </AccordionDetails>
        </Accordion>
      </Box>
    </Box>
  )

  const renderResults = () => {
    if (isLoading) {
      return (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )
    }

    if (error) {
      return (
        <Typography color="error">
          Error loading results. Please try again.
        </Typography>
      )
    }

    const unpaginatedResults = data?.results || []

    return (
      <Box>
        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
        {unpaginatedResults.map((result: any, index: number) => (
          <Card
            variant="outlined"
            sx={{ mb: 2, p: 2 }}
            key={`${result.id}-${index}`}
          >
            <Grid container spacing={2}>
              <Grid size={9}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mb: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    fontWeight="bold"
                  >
                    {result.resource_type || "Video"}
                  </Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: "bold", mb: 2 }}>
                  <Link
                    href={result.url || "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    color="inherit"
                    underline="hover"
                  >
                    {result.title}
                  </Link>
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 3 }}
                >
                  Format:{" "}
                  {result.delivery
                    ?.map((d: { code: string }) => d.code)
                    .join(", ") || "Online"}
                </Typography>

                <Box sx={{ display: "flex", gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    sx={{
                      borderRadius: "50%",
                      minWidth: "40px",
                      width: "40px",
                      p: 0,
                    }}
                  >
                    <RiListCheck size={18} />
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    sx={{
                      borderRadius: "50%",
                      minWidth: "40px",
                      width: "40px",
                      p: 0,
                    }}
                  >
                    <RiBookmarkLine size={18} />
                  </Button>
                </Box>
              </Grid>
              <Grid
                size={3}
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-end",
                  justifyContent: "space-between",
                }}
              >
                <Typography variant="caption" fontWeight="bold">
                  {result.free
                    ? "Free"
                    : result.prices?.length
                      ? `$${result.prices[0]}`
                      : ""}
                </Typography>
                {result.image && result.image.url && (
                  <Box
                    component="img"
                    src={result.image.url}
                    alt={result.title}
                    sx={{
                      width: "100%",
                      height: "100px",
                      objectFit: "cover",
                      borderRadius: 1,
                    }}
                  />
                )}
              </Grid>
            </Grid>
          </Card>
        ))}
        {unpaginatedResults.length === 0 && (
          <Typography sx={{ mt: 2 }}>
            No results found for "{searchQuery}"
          </Typography>
        )}
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", p: 2 }}>
      <Box
        sx={{ mb: 4, px: 20, display: "flex", gap: 2, alignItems: "center" }}
      >
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
        <form
          style={{ flexGrow: 1 }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              setSearchQuery(inputValue)
            }
          }}
        >
          <SearchInput
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onSubmit={handleSubmit}
            onClear={() => {
              setInputValue("")
              setSearchQuery("")
            }}
            placeholder="Search for courses, programs, and learning materials..."
            fullWidth
            size="large"
          />
        </form>
        <FormControlLabel
          control={
            <Checkbox
              checked={hybridSearch}
              onChange={(e) => setHybridSearch(e.target.checked)}
              color="primary"
            />
          }
          label="Hybrid Search"
          sx={{ ml: 1, whiteSpace: "nowrap" }}
        />
      </Box>

      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 3 }}>{renderFilterSidebar()}</Grid>

        <Grid size={{ xs: 12, md: 9 }}>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              mb: 3,
            }}
          >
            <ToggleButtonGroup
              value={resourceTypeGroup}
              exclusive
              onChange={(e, val) => {
                if (val) setResourceTypeGroup(val)
              }}
              size="small"
              sx={{ bgcolor: "#f5f5f5" }}
            >
              <ToggleButton value="all" sx={{ textTransform: "none", px: 2 }}>
                All
              </ToggleButton>
              <ToggleButton
                value="course"
                sx={{ textTransform: "none", px: 2 }}
              >
                Courses
              </ToggleButton>
              <ToggleButton
                value="program"
                sx={{ textTransform: "none", px: 2 }}
              >
                Programs
              </ToggleButton>
              <ToggleButton
                value="learning_material"
                sx={{ textTransform: "none", px: 2 }}
              >
                Learning Materials
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {renderResults()}
        </Grid>
      </Grid>
    </Box>
  )
}

export default LearningResourcesSearchContent
