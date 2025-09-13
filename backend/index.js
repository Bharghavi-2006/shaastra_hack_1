const express = require("express");
const cors = require("cors");
const multer = require("multer");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
  res.json({ message: "Backend running 🚀" });
});

app.post("/api/verify", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ status: "error", message: "No file uploaded" });
  }

  const fileName = req.file.originalname.toLowerCase();
  let result = { status: "valid", message: "Document looks valid (dummy check)" };

  const invalidPatterns = [
    "passbook", "bill", "receipt", "business card", "scenery",
    "selfie", "animal", "back", "qr", "cropped", "blank"
  ];

  for (let pattern of invalidPatterns) {
    if (fileName.includes(pattern)) {
      result = { status: "not valid", reason: "Invalid document" };
      break;
    }
  }

  res.json(result);
});

app.listen(PORT, () => console.log(`✅ Backend running at http://localhost:${PORT}`));
