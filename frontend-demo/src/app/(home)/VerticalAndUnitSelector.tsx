import { useEffect, useState } from "react"
import { UseQueryResult } from "@tanstack/react-query"
import TextField from "@mui/material/TextField"
import { CourseV2BlocksResponse } from "@/services/openedx/client"
import Link from "@mui/material/Link"
import MenuItem from "@mui/material/MenuItem"

const validateVerticalBlockId = (value: string) => {
  if (!value.startsWith("block-v1:")) {
    throw new Error("Block ID must start with 'block-v1:'")
  }
  if (!value.includes("+type@vertical+block@")) {
    throw new Error("Block ID must contain '+type@vertical+block@'")
  }
  if (value.split("@").length !== 3) {
    throw new Error("Block ID must contain exactly two '@'")
  }

  // Good enough for us.
  return true
}
const isVerticalBlockId = (value: string) => {
  try {
    validateVerticalBlockId(value)
    return true
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    return false
  }
}

/**
 * Extract the vertical block id, assuming the input is EITHER a vertical block id
 * OR a URL containing a vertical block id in its path.
 *
 * An example vertical URL is:
 *  origina:
 *    https://courses-qa.mitxonline.mit.edu/learn/course/course-v1:MITxT+3.012Sx+3T2024/block-v1:MITxT+3.012Sx+3T2024+type@sequential+block@097bd07832d9455998a78f8d1dcca4be/block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@027c11b88c9b4f028127ef968b58c5b9
 *  split across new lines:
 *   https://courses-qa.mitxonline.mit.edu/learn/course \
 *    /course-v1:MITxT+3.012Sx+3T2024 \
 *    /block-v1:MITxT+3.012Sx+3T2024+type@sequential+block@097bd07832d9455998a78f8d1dcca4be \
 *    /block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@027c11b88c9b4f028127ef968b58c5b9
 */
const extractVerticalBlockId = (
  value: string,
): { blockId: string | null; errMsg: string | null } => {
  if (!value) {
    return { blockId: null, errMsg: "Vertical ID is required." }
  }
  try {
    validateVerticalBlockId(value)
    return { blockId: value, errMsg: null }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    // pass
  }
  try {
    const urlObj = new URL(value)
    const blockId = urlObj.pathname.split("/").at(-1) ?? ""
    validateVerticalBlockId(blockId)
    return { blockId, errMsg: null }
  } catch (err) {
    if (err instanceof Error) {
      return { blockId: null, errMsg: err.message }
    }
  }
  return { blockId: null, errMsg: "Could not extract block id" }
}

const getVerticalHelpText = (
  overridingErrMsg: string | null,
  vertical: UseQueryResult<CourseV2BlocksResponse>,
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

type VerticalAndUnitSelectorProps = {
  verticalSettingsName: string
  unitSettingsName: string
  unitFieldLabel: string
  unitFilterType?: string
  vertical: UseQueryResult<CourseV2BlocksResponse, Error>
  settings: Record<string, string>
  setSettings: (settings: Record<string, string>) => void
}

/**
 * A react component that
 * - allows selecting an OpenEdx vertical block
 * - queries the vertical for its children
 * - allows selecting a unit from the children
 * - syncs the selected vertical and unit with the URL
 */
const VerticalAndUnitSelector: React.FC<VerticalAndUnitSelectorProps> = ({
  verticalSettingsName,
  unitSettingsName,
  unitFieldLabel,
  unitFilterType,
  vertical,
  settings,
  setSettings,
}) => {
  const [verticalParseError, setVerticalParseError] = useState<string | null>(
    extractVerticalBlockId(settings[verticalSettingsName]).errMsg,
  )

  const units = Object.values(vertical.data?.blocks ?? {}).filter((block) => {
    if (block.id === vertical.data?.root) return false
    return unitFilterType ? block.type === unitFilterType : true
  })
  const selectedUnitId = units.find(
    (block) => block.id === settings[unitSettingsName],
  )?.id
  useEffect(() => {
    if (!selectedUnitId && vertical.data) {
      setSettings({ [unitSettingsName]: "" })
    }
  }, [selectedUnitId, setSettings, unitSettingsName, vertical.data])

  const [verticalValue, setVerticalValue] = useState<string>(
    settings[verticalSettingsName],
  )
  useEffect(() => {
    /**
     * Track verticalValue locally and sync with URL rather than using URL
     * value directly to keep the input updates synchhronous. Otherwise you run
     * into https://stackoverflow.com/questions/46000544/react-controlled-input-cursor-jumps
     */
    setSettings({ [verticalSettingsName]: verticalValue })
  }, [verticalValue, setSettings, verticalSettingsName])

  console.log(vertical)

  return (
    <>
      <TextField
        size="small"
        label="Openedx Unit (vertical) ID or URL"
        fullWidth
        autoCapitalize="off"
        spellCheck={false}
        multiline
        value={verticalValue}
        onChange={(e) => {
          const { blockId, errMsg } = extractVerticalBlockId(e.target.value)
          setVerticalParseError(errMsg)
          setVerticalValue(blockId ?? e.target.value)
        }}
        required
        error={!!verticalParseError || vertical.isError}
        helperText={getVerticalHelpText(verticalParseError, vertical)}
      />
      <TextField
        label={unitFieldLabel}
        size="small"
        fullWidth
        margin="normal"
        select
        required
        /**
         * Avoid passing an out-of-range value while the options are loading.
         */
        value={selectedUnitId ?? ""}
        onChange={(e) => {
          setSettings({ [unitSettingsName]: e.target.value })
        }}
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
    </>
  )
}

export default VerticalAndUnitSelector
export { isVerticalBlockId }
