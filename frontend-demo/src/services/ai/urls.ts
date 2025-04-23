const BASE_URL = process.env.NEXT_PUBLIC_MITOL_API_BASE_URL ?? ""

const RECOMMENDATION_GPT_URL = `${BASE_URL}/http/recommendation_agent/`
const SYLLABUS_GPT_URL = `${BASE_URL}/http/syllabus_agent/`
const VIDEO_GPT_URL = `${BASE_URL}/http/video_gpt_agent/`
const ASSESSMENT_GPT_URL = `${BASE_URL}/http/tutor_agent/`

export {
  RECOMMENDATION_GPT_URL,
  SYLLABUS_GPT_URL,
  VIDEO_GPT_URL,
  ASSESSMENT_GPT_URL,
}
