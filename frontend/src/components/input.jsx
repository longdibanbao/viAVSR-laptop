import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import Webcam from 'react-webcam';
import { useRef, useCallback } from "react"
//
// const webcam = () => (
//   <Webcam />
// );
//

const VideoInput = () => {


  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  // conse function

  // const 




  return (
    <div className='m-10 border border-black'>
      <Webcam
        mirrored={true}
      />
    </div>
  )
}
export default VideoInput;
