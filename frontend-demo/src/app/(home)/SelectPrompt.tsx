import TextareaAutosize from "@mui/material/TextareaAutosize"
import type { TextareaAutosizeProps } from "@mui/material/TextareaAutosize"
import { promptQueries } from "@/services/ai"
import { useQuery } from "@tanstack/react-query"

const SelectPrompt: React.FC<
  TextareaAutosizeProps & {
    value: string
  }
> = (props) => {
  const prompt = useQuery(promptQueries.get("recommendation"))
  return (
    <TextareaAutosize
      aria-label="System Prompt"
      {...props}
      /**
       * Avoid passing an out-of-range value while the options are loading.
       */
      value={prompt.isLoading ? "" : props.value}
    />
  )
}

export default SelectPrompt
