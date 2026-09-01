import { useRef, useState } from "react";
import {
  Upload,
  File as FileIcon,
  Trash2,
} from "lucide-react";

const VideoUploader = () => {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("pending");
  const [result, setResult] = useState(null);

  // User chọn file
  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0] || null;

    setFile(selectedFile);
    setStatus("pending");
    setResult(null);
  };

  // Upload video lên API
  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");

    try {
      const formData = new FormData();

      formData.append("video", file);

      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      console.log("API response:", data);

      setResult(data);
      setStatus("success");
    } catch (error) {
      console.error(error);
      setStatus("error");
    }
  };

  // Xóa file
  const handleRemoveFile = () => {
    setFile(null);
    setStatus("pending");
    setResult(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setStatus("pending");
    setResult(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleUpload();
      }}
      onReset={handleReset}
      className="mx-auto w-full max-w-xl space-y-5"
    >
      {/* Heading */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Upload Video
        </h1>

        <p className="mt-1 text-sm text-gray-500">
          Upload a video to run the model.
        </p>
      </div>

      {/* Upload box */}
      <div
        onClick={() => inputRef.current?.click()}
        className="
          flex
          h-64
          cursor-pointer
          items-center
          justify-center
          rounded-xl
          border-2
          border-dashed
          border-gray-300
          bg-gray-50
          transition
          hover:border-blue-400
          hover:bg-blue-50
        "
      >
        <div className="text-center">

          {/* Upload icon */}
          <div
            className="
              mx-auto
              mb-4
              flex
              h-16
              w-16
              items-center
              justify-center
              rounded-full
              bg-blue-100
              text-blue-600
            "
          >
            <Upload size={32} />
          </div>

          <h2 className="text-lg font-semibold text-gray-800">
            Drop your video here
          </h2>

          <p className="mt-1 text-sm text-gray-500">
            or click to browse
          </p>

          <p className="mt-3 text-xs text-gray-400">
            Supported formats: MP4, WebM, MOV
          </p>
        </div>
      </div>

      {/* Hidden input */}
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={handleFileChange}
      />

      {/* File information */}
      {file && (
        <div className="space-y-4">

          <div
            className="
              flex
              items-center
              gap-4
              rounded-xl
              border
              border-gray-200
              bg-white
              p-4
              shadow-sm
            "
          >
            {/* File icon */}
            <div
              className="
                flex
                h-12
                w-12
                shrink-0
                items-center
                justify-center
                rounded-lg
                bg-gray-100
                text-gray-600
              "
            >
              <FileIcon size={26} />
            </div>

            {/* File details */}
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-gray-900">
                {file.name}
              </p>

              <p className="mt-1 text-sm text-gray-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>

            {/* Remove button */}
            {status === "pending" && (
              <button
                type="button"
                onClick={handleRemoveFile}
                className="
                  rounded-lg
                  p-2
                  text-gray-400
                  transition
                  hover:bg-red-50
                  hover:text-red-500
                "
              >
                <Trash2 size={22} />
              </button>
            )}

            {/* Uploading */}
            {status === "uploading" && (
              <span className="text-sm font-medium text-blue-600">
                Uploading...
              </span>
            )}

            {/* Success */}
            {status === "success" && (
              <span className="text-sm font-medium text-green-600">
                Success
              </span>
            )}

            {/* Error */}
            {status === "error" && (
              <span className="text-sm font-medium text-red-600">
                Failed
              </span>
            )}
          </div>

          {/* Upload / Reset button */}
          {status === "success" ? (
            <button
              type="reset"
              className="
                w-full
                rounded-lg
                bg-gray-900
                px-5
                py-3
                font-medium
                text-white
                transition
                hover:bg-gray-800
              "
            >
              Upload Another Video
            </button>
          ) : (
            <button
              type="submit"
              disabled={status === "uploading"}
              className="
                w-full
                rounded-lg
                bg-blue-600
                px-5
                py-3
                font-medium
                text-white
                transition
                hover:bg-blue-700
                disabled:cursor-not-allowed
                disabled:opacity-50
              "
            >
              {status === "uploading"
                ? "Processing..."
                : "Upload Video"}
            </button>
          )}

          {/* Model response */}
          {result && (
            <div
              className="
                rounded-xl
                border
                border-gray-200
                bg-gray-50
                p-4
              "
            >
              <h2 className="mb-2 font-semibold text-gray-900">
                Model Response
              </h2>

              <pre className="overflow-auto text-sm text-gray-700">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}

        </div>
      )}
    </form>
  );
};

export default VideoUploader;
