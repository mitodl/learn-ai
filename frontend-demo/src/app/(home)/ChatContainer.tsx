import Typography from "@mui/material/Typography"
import { styled } from "@mui/material/styles"

const Container = styled("div")({
  position: "relative",
  minHeight: "600px",
})

const Overlay = styled("div")({
  position: "absolute",
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: "rgba(0, 0, 0, 0.1)",
  backdropFilter: "blur(0.75px)",
  zIndex: 1000,
})

const ChatContainer: React.FC<
  React.DOMAttributes<HTMLDivElement> & {
    enabled: boolean
  }
> = ({ enabled, ...props }) => {
  return (
    <Container {...props} inert={!enabled}>
      {props.children}
      {!enabled && <Overlay />}
      {!enabled && (
        <Typography
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            zIndex: 1000,
            transform: "translate(-50%, -50%)",
            maxWidth: "33%",
            textAlign: "center",
          }}
          color="primary"
        >
          Some required fields are missing or invalid.
        </Typography>
      )}
    </Container>
  )
}

export default ChatContainer
