import react from "react"

// import { Dice1 } from "lucide-react"

import "./index.css"



import ItemBorder from "./components/border.jsx"
import MyHeader from "./components/header.jsx"
import ControlAndStatus from "./components/BottomBar.jsx"
import VideoInput from "./components/input.jsx"

import VideoUploader from "./components/uploadVideo.jsx"

const App = () => {
  return (
    <div className="p-[5px]">
      <MyHeader />
      <div className="flex w-full">
        <ItemBorder>
          <div>
            {/* <VideoInput /> */}
            <VideoUploader />
          </div>
        </ItemBorder>
        <ItemBorder><div>Processed</div></ItemBorder>
        <ItemBorder><div>Result</div></ItemBorder>
      </div>

      <ItemBorder className="!h-auto min-h-[100px]">
        <ControlAndStatus />
      </ItemBorder>

    </div>
  )
}


export default App

