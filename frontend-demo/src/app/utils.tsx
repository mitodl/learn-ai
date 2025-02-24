/**
 * Extracts a JSON object from a comment string
 * @param comment the comment string
 * @returns the JSON object
 */
export const extractJSONFromComment = (comment: string) => {
  const jsonStr = comment.toString().match(/<!-{2}(.*)-{2}>/)?.[1] || "{}"
  try {
    return JSON.parse(jsonStr)
  } catch (e) {
    console.error("error parsing JSON from comment", comment, e)
    return {}
  }
}
