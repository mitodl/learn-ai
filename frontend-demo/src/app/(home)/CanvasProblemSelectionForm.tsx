import React from "react"
import TextField from "@mui/material/TextField"
import { useFormik } from "formik"



type CanvasProblemSelectionFormProps = {
  selectedRun: string
  selectedProblemSet: string
  defaultRun: string
  defaultProblemSet: string
  onSubmit: (values: { run: string; problem_set: string }) => void
  onReset: () => void
}

const CanvasProblemSelectionForm: React.FC<CanvasProblemSelectionFormProps> = ({
  selectedRun,
  selectedProblemSet,
  defaultRun,
  defaultProblemSet,
  onSubmit,
  onReset,
}) => {
  const [editing, setEditing] = React.useState(false)

  const formik = useFormik({
    enableReinitialize: true,
    initialValues: {
      run: defaultRun,
      problem_set: defaultProblemSet,
    },
    onSubmit: (values) => {
      setEditing(false)
      onSubmit(values)
    },
    validateOnChange: false,
  })


  return (
    <form onSubmit={formik.handleSubmit}>
      <TextField
        size="small"
        label="Course Run Readable ID"
        fullWidth
        autoCapitalize="off"
        spellCheck={false}
        margin="normal"
        name="run"
        value={formik.values.run}
        onChange={formik.handleChange}
        required
      />
    </form>
  )
}

export default CanvasProblemSelectionForm
