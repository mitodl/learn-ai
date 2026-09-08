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
      <MenuItem key="traditional" value="elastic">
        Elasticsearch
      </MenuItem>
      <MenuItem key="vector" value="vector">
        Vector
      </MenuItem>
    </TextField>
  )
}

export default SelectSearchURL
