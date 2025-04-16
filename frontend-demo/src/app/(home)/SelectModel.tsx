import MenuItem from "@mui/material/MenuItem"
import TextField from "@mui/material/TextField"
import type { TextFieldProps } from "@mui/material/TextField"
import { llmModelsQueries } from "@/services/ai"
import { useQuery } from "@tanstack/react-query"

const SelectModel: React.FC<
  TextFieldProps & {
    value: string
  }
> = (props) => {
  const llmModels = useQuery(llmModelsQueries.list())
  return (
    <TextField
      label="Model"
      size="small"
      fullWidth
      select
      {...props}
      /**
       * Avoid passing an out-of-range value while the options are loading.
       */
      value={llmModels.isLoading ? "" : props.value}
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
