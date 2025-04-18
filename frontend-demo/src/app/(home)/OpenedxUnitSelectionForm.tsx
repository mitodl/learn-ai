import React from "react"
import { UseQueryResult } from "@tanstack/react-query"
import TextField from "@mui/material/TextField"
import { CourseV2BlocksResponse } from "@/services/openedx/client"
import Link from "@mui/material/Link"
import MenuItem from "@mui/material/MenuItem"
import { useV2Block } from "@/services/openedx"
import * as yup from "yup"
import { useFormik } from "formik"
import Alert from "@mui/material/Alert"
import Stack from "@mui/material/Stack"
import { Button } from "@mitodl/smoot-design"

const verticalSchema = yup
  .string()
  .required()
  .transform((value) => {
    // If value is a URL, except a module_id, try to extract the vertical module id.

    // edx_module_ids are valid URLs I guess, so exclude these
    if (value.startsWith("block-v1:")) return value
    try {
      const urlObj = new URL(value)
      const blockId = urlObj.pathname.split("/").at(-1)
      if (blockId) {
        return blockId
      }
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (err) {
      // pass
    }
    return value
  })
  .test((value, ctx) => {
    if (!value.startsWith("block-v1:")) {
      return ctx.createError({
        message: "Block ID must start with 'block-v1:'",
      })
    }
    if (!value.includes("+type@vertical+block@")) {
      return ctx.createError({
        message: "Block ID must contain '+type@vertical+block@'",
      })
    }
    if (value.split("@").length !== 3) {
      return ctx.createError({
        message: "Block ID must contain exactly two '@'",
      })
    }

    // Good enough for us.
    return true
  })

const getVerticalHelpText = (
  vertical: UseQueryResult<CourseV2BlocksResponse>,
  overridingErrMsg?: string,
) => {
  if (overridingErrMsg) {
    return overridingErrMsg
  }
  if (vertical.isError) {
    return "Vertical not found."
  }
  if (vertical.data) {
    const root = vertical.data.blocks[vertical.data.root]
    return (
      <span>
        Found vertical{" "}
        <Link target="_blank" href={root.lms_web_url}>
          {root.display_name}
        </Link>
      </span>
    )
  }
}

const ChosenUnitDisplay = ({
  verticalId,
  unitId,
  unitLabel,
  onEdit,
}: {
  verticalId: string
  unitId: string
  unitLabel: string
  onEdit: () => void
}) => {
  const vertical = useV2Block({
    blockUsageKey: verticalId,
  })

  const unit = vertical.data?.blocks[unitId]

  return (
    <Stack>
      {vertical.isError && (
        <Alert severity="error">Error loading vertical data.</Alert>
      )}
      {vertical.data && !unit && (
        <Alert severity="error">
          Could not find unit with ID {unitId} in vertical {verticalId}
        </Alert>
      )}
      <Stack direction="row" gap="16px">
        <TextField
          label={`Selected ${unitLabel}`}
          sx={{ flex: 1, minWidth: "0px" }}
          size="small"
          slotProps={{
            input: { readOnly: true },
            inputLabel: { shrink: true },
          }}
          value={vertical.isLoading ? "Loading..." : (unit?.display_name ?? "")}
        />
        <Button variant="secondary" onClick={onEdit}>
          Edit
        </Button>
      </Stack>
      {unit && (
        <Link
          sx={(theme) => ({
            marginTop: "8px",
            wordBreak: "break-all",
            typography: theme.typography.body3,
          })}
          href={unit.lms_web_url}
          target="_blank"
        >
          {unit.lms_web_url}
        </Link>
      )}
    </Stack>
  )
}

type OpenedxUnitSelectionFormProps = {
  selectedVertical: string
  selectedUnit: string
  defaultVertical: string
  defaultUnit: string
  unitFilterType: string
  onSubmit: (values: { vertical: string; unit: string }) => void
  onReset: () => void
  unitLabel: string
}

const schema = yup.object().shape({
  vertical: verticalSchema,
  unit: yup.string().required(),
})

const OpenedxUnitSelectionForm: React.FC<OpenedxUnitSelectionFormProps> = ({
  selectedUnit,
  selectedVertical,
  defaultVertical,
  defaultUnit,
  unitFilterType,
  onSubmit,
  onReset,
  unitLabel,
}) => {
  const [editing, setEditing] = React.useState(false)

  const formik = useFormik({
    enableReinitialize: true,
    initialValues: {
      vertical: defaultVertical,
      unit: defaultUnit,
    },
    validationSchema: schema,
    onSubmit: (values) => {
      setEditing(false)
      onSubmit(values)
    },
    validateOnChange: false,
  })

  const verticalQuery = useV2Block(
    {
      blockUsageKey: formik.values.vertical,
    },
    {
      enabled: !formik.errors.vertical,
    },
  )

  const units = Object.values(verticalQuery.data?.blocks ?? {}).filter(
    (block) => {
      if (block.id === verticalQuery.data?.root) return false
      return unitFilterType ? block.type === unitFilterType : true
    },
  )

  if (!editing) {
    return (
      <ChosenUnitDisplay
        unitLabel={unitLabel}
        verticalId={selectedVertical}
        unitId={selectedUnit}
        onEdit={() => setEditing(true)}
      />
    )
  }

  return (
    <form onSubmit={formik.handleSubmit}>
      <TextField
        size="small"
        label="Openedx Unit (vertical) ID or URL"
        fullWidth
        autoCapitalize="off"
        spellCheck={false}
        multiline
        margin="normal"
        name="vertical"
        value={formik.values.vertical}
        onChange={async (e) => {
          await formik.setFieldValue(
            "vertical",
            verticalSchema.cast(e.target.value),
            true,
          )
          // The third arg here is supposed to stop validation
          // but that doesn't seem to be working. Unsure why.
          // For now, just manually unset the unit error.
          await formik.setFieldValue("unit", "", false)
          formik.setFieldError("unit", undefined)
        }}
        required
        error={Boolean(
          formik.errors.vertical || verticalQuery.isError, // API error,
        )}
        helperText={getVerticalHelpText(verticalQuery, formik.errors.vertical)}
      />
      <TextField
        label={unitLabel}
        size="small"
        fullWidth
        margin="normal"
        select
        name="unit"
        value={formik.values.unit}
        onChange={formik.handleChange}
        error={!!formik.errors.unit}
        helperText={formik.errors.unit}
      >
        <MenuItem value="">Select one</MenuItem>
        {units.map((block) => {
          return (
            <MenuItem key={block.id} value={block.id}>
              {block.display_name}
            </MenuItem>
          )
        })}
      </TextField>
      <Stack direction="row" gap="16px">
        <Button
          variant="secondary"
          onClick={() => {
            onReset()
            formik.resetForm()
            setEditing(false)
          }}
        >
          Reset
        </Button>
        <Button type="submit">Submit</Button>
      </Stack>
    </form>
  )
}

export default OpenedxUnitSelectionForm
export { verticalSchema }
