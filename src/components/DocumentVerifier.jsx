import { useState } from "react";

function DocumentVerifier() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload a file first!");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/api/verify", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ status: "error", message: "Something went wrong!" });
    }

    setLoading(false);
  };

  return (
    <div className="document-verifier">
      <h2 className="title">Document Verifier</h2>

      <input
        type="file"
        accept=".jpg,.jpeg,.png"
        onChange={handleFileChange}
        className="file-input"
      />

      <p className="note-text">
        ⚠️ Please upload the images in jpg or png format only.
      </p>

      {preview && (
        <div className="preview-section">
          <p className="preview-label">Preview:</p>
          <img
            src={preview}
            alt="Uploaded Preview"
            className="preview-image"
          />
        </div>
      )}

      <button
        onClick={handleUpload}
        className="verify-button"
        disabled={loading}
      >
        {loading ? "Verifying..." : "Verify Document"}
      </button>

      {result && (
        <div className="result-section">
          <h3 className="result-title">Result:</h3>
          <pre className="result-content">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default DocumentVerifier;
