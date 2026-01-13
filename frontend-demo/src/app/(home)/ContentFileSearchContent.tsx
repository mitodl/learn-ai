import { VectorContenfilesQueries } from "@/services/learn"
import { useQuery } from "@tanstack/react-query"
import React, { useState } from "react"
import {
  SearchInput,
  type SearchInputProps,
} from "@/components/SearchInput/SearchInput"
import { Card, Typography, CircularProgress, Link } from "@mui/material"
import Grid from "@mui/material/Grid2"
import Box from "@mui/material/Box"

const ContentFileSearchContent: React.FC = () => {
  const [inputValue, setInputValue] = useState("")
  const [searchQuery, setSearchQuery] = useState("")

  const { data, isLoading, error } = useQuery({
    ...VectorContenfilesQueries.listing({ q: searchQuery, group_by: "key" }),
    enabled: !!searchQuery,
  })

  const handleSubmit: SearchInputProps["onSubmit"] = (e) => {
    e.preventDefault()
    setSearchQuery(inputValue)
  }

  const handleClear = () => {
    setInputValue("")
    setSearchQuery("")
  }

  return (
    <Box sx={{ maxWidth: 960, mx: "auto", p: 2 }}>
      <Typography variant="h3" gutterBottom>
        Vector Content File Search
      </Typography>

      <Box sx={{ mb: 4 }}>
        <SearchInput
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onSubmit={handleSubmit}
          onClear={handleClear}
          placeholder="Search for content..."
          fullWidth
          size="medium"
        />
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
        {data?.results?.map((result) => (
          <Grid size={{ xs: 12 }} key={result.id}>
            <Card variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" component="div">
                <Link
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {result.content_title || "Untitled"}
                </Link>
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {result.excerpt ||
                  result.description ||
                  "No description available."}
              </Typography>
              {result.content_title && (
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Source: {result.content_title}
                </Typography>
              )}
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
