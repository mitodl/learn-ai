import Link from "next/link"

export const metadata = {
  title: "Home",
}

const Page = () => {
  return (
    <div>
      <ul>
        <li>
          <Link href="/http">Http Chat</Link>
        </li>
        <li>
          <Link href="/ws">Websocket Chat</Link>
        </li>
      </ul>
    </div>
  )
}

export default Page
