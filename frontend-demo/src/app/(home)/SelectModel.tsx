import MenuItem from "@mui/material/MenuItem"
import TextField from "@mui/material/TextField"
import { llmModelsQueries } from "@/services/ai"
import { useQuery } from "@tanstack/react-query"
import { useState } from "react"

type SelectModelProps = {
  onChange: (model?: string) => void
}
const SelectModel = ({ onChange }: SelectModelProps) => {
  const [value, setValue] = useState<string>("")
  const llmModels = useQuery(llmModelsQueries.list())
  return (
    <TextField
      label="Model"
      size="small"
      fullWidth
      select
      value={value}
      onChange={(e) => {
        const value = e.target.value
        setValue(value)
        if (value) {
          onChange(e.target.value)
        } else {
          onChange(undefined)
        }
      }}
    >
      <MenuItem value="">Model</MenuItem>
      {llmModels.data?.map((model) => (
        <MenuItem key={model.litellm_id} value={model.litellm_id}>
          {model.name}
        </MenuItem>
      ))}
    </TextField>
  )
}

export default SelectModel
