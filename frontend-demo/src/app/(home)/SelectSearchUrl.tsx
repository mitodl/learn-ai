import MenuItem from "@mui/material/MenuItem"
import TextField from "@mui/material/TextField"
import type { TextFieldProps } from "@mui/material/TextField"

const SelectSearchURL: React.FC<
  TextFieldProps & {
    value: string
  }
> = (props) => {
  return (
    <TextField
      label="Search Type"
      size="small"
      margin="normal"
      fullWidth
      select
      {...props}
      /**
       * Avoid passing an out-of-range value while the options are loading.
       */
      value={props.value}
    >
      <MenuItem value="">Search Type</MenuItem>
      <MenuItem
        key="traditional"
        value={process.env.NEXT_PUBLIC_MIT_SEARCH_ELASTIC_URL}
      >
        Elasticsearch
      </MenuItem>
      <MenuItem
        key="vector"
        value={process.env.NEXT_PUBLIC_MIT_SEARCH_VECTOR_URL}
      >
        Vector
      </MenuItem>
    </TextField>
  )
}

export default SelectSearchURL
