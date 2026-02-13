import { VectorContenfilesQueries } from "@/services/learn"
import { useQuery } from "@tanstack/react-query"
import React, { useState } from "react"
import {
  SearchInput,
  type SearchInputProps,
} from "@/components/SearchInput/SearchInput"
import {
  Card,
  Typography,
  CircularProgress,
  Link,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  type SelectChangeEvent,
} from "@mui/material"
import Grid from "@mui/material/Grid2"
import Box from "@mui/material/Box"

import {
  RiBookOpenFill,
  RiCodeSSlashFill,
  RiFileTextFill,
  RiGraduationCapFill,
  RiInformationFill,
  RiListOrdered,
  RiListUnordered,
  RiQuestionMark,
  RiSplitCellsHorizontal,
  RiVidicon2Fill,
} from "@remixicon/react"

const allowedTypes = [
  "chapter",
  "course",
  "html",
  "info",
  "problem",
  "sequential",
  "static",
  "vertical",
  "video",
]

const getContentType = (sourcePath: string) => {
  if (sourcePath.endsWith(".srt")) {
    return "video"
  }
  return sourcePath ? sourcePath.split("/")[1] : "static"
}

const getContentTypeIcon = (type: string) => {
  switch (type) {
    case "chapter":
      return <RiBookOpenFill />
    case "course":
      return <RiGraduationCapFill />
    case "html":
      return <RiCodeSSlashFill />
    case "info":
      return <RiInformationFill />
    case "problem":
      return <RiQuestionMark />
    case "sequential":
      return <RiListOrdered />
    case "split_test":
      return <RiSplitCellsHorizontal />
    case "static":
      return <RiFileTextFill />
    case "vertical":
      return <RiListUnordered />
    case "video":
      return <RiVidicon2Fill />
    default:
      return <RiFileTextFill />
  }
}

const ContentFileSearchContent: React.FC = () => {
  const [inputValue, setInputValue] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [platform, setPlatform] = useState("")

  const { data, isLoading, error } = useQuery({
    ...VectorContenfilesQueries.listing({
      q: searchQuery,
      group_by: "key",
      group_size: 1,
      platform: platform || undefined,
    }),
    enabled: !!searchQuery,
  })

  const handleSubmit: SearchInputProps["onSubmit"] = (e) => {
    e.preventDefault()
    setSearchQuery(inputValue)
  }

  const handleClear = () => {
    setInputValue("")
    setSearchQuery("")
    setPlatform("")
  }

  const handlePlatformChange = (event: SelectChangeEvent) => {
    setPlatform(event.target.value)
  }

  return (
    <Box sx={{ maxWidth: 960, mx: "auto", p: 2 }}>
      <Typography variant="h3" gutterBottom>
        Vector Based Content File Search
      </Typography>

      <Box sx={{ mb: 4, display: "flex", gap: 2 }}>
        <SearchInput
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onSubmit={handleSubmit}
          onClear={handleClear}
          placeholder="Search for content..."
          fullWidth
          size="large"
        />
        <Box sx={{ minWidth: 200 }}>
          <FormControl fullWidth>
            <InputLabel id="platform-select-label">Platform</InputLabel>
            <Select
              labelId="platform-select-label"
              id="platform-select"
              value={platform}
              label="Platform"
              onChange={handlePlatformChange}
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              <MenuItem value="mitxonline">MITx Online</MenuItem>
              <MenuItem value="xpro">xPRO</MenuItem>
              <MenuItem value="edx">edX</MenuItem>
              <MenuItem value="ocw">OpenCourseWare</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {isLoading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Typography color="error">
          Error loading results. Please try again.
        </Typography>
      )}

      <Grid container spacing={2}>
        {data?.results
          ?.filter((result) => {
            if (!result.title || !result.url || !result.source_path)
              return false
            const type = getContentType(result.source_path)
            return allowedTypes.includes(type)
          })
          .map((result) => (
            <Grid size={{ xs: 12 }} key={result.id}>
              <Card variant="outlined" sx={{ p: 2 }}>
                <Box display="flex" alignItems="center">
                  <Box color="text.secondary" sx={{ paddingRight: "30px" }}>
                    {getContentTypeIcon(getContentType(result.source_path))}
                  </Box>
                  <Box>
                    <Typography variant="body3" color="text.primary">
                      {result.platform?.name}
                    </Typography>
                    <Typography variant="body1" component="div">
                      <Link
                        href={result.url || ""}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {result.title}
                      </Link>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {result.run_title}
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
          ))}
        {data?.results?.length === 0 && (
          <Typography>No results found for "{searchQuery}"</Typography>
        )}
      </Grid>
    </Box>
  )
}

export default ContentFileSearchContent
