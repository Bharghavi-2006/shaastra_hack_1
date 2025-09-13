const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const FormData = require("form-data");
const axios = require("axios");

const app = express();
app.use(cors());

// Multer: store uploaded files temporarily
const upload = multer({ dest: "tmp_uploads/" });

app.post("/api/verify-id", upload.single("image"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    const flaskRes = await axios.post("http://localhost:5000/api/verify", form, {
      headers: form.getHeaders(),
    });

    // cleanup temp file
    fs.unlink(req.file.path, () => {});

    // forward only "valid" (or whole response if you want)
    res.json({ valid: flaskRes.data.valid, details: flaskRes.data });
  } catch (err) {
    console.error("Error verifying:", err.message);
    res.status(500).json({ error: "Verification failed" });
  }
});

app.listen(3000, () => {
  console.log("Node backend running on http://localhost:3000");
});
