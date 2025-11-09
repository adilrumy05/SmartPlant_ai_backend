import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import cors from 'cors';
import { // Import helper functions from db.js for database operations
  getOrCreateSpeciesId,
  insertObservation,
  insertAiResults,
  getObservationWithResults,
} from './db.js';

// Environment setup and Express app configuration
const __filename = fileURLToPath(import.meta.url); // Get the current file path 
const __dirname = path.dirname(__filename); // Get the current directory path
const app = express(); // Initialize the Express application

// Python worker, which loads the AI model once and handles all inferences
let pyWorker = null; // Will store the persistent Python process reference

function getWorker() { // Create or return an existing worker process
  if (pyWorker && !pyWorker.killed) return pyWorker; // Reuse existing worker if alive

  // Get Python executable path and worker script path
  const pythonPath = process.env.PYTHON_PATH || 'python';
  const workerPath = path.join(__dirname, '..', 'python', 'worker.py');

  
  pyWorker = spawn(pythonPath, [workerPath], { stdio: ['pipe', 'pipe', 'pipe'] }); // Spawn a new python worker process
  pyWorker.stderr.on('data', d => console.error('[pyworker]', String(d).trim())); // Log any stderr output from the worker
  pyWorker.on('exit', (code) => { // If worker exits, log and reset reference so it can restart
    console.error('[pyworker] exited with code', code);
    pyWorker = null;
  });
  return pyWorker;
}

// Send an inference request to the worker and wait for response
function inferWithWorker(imagePath, topk = 5) {
  return new Promise((resolve, reject) => {
    const worker = getWorker(); // Ensure we have a worker process
    let buf = ''; // Buffer to accumulate stdout data

    // Listen for worker output
    const onData = (d) => {
      buf += d.toString();
      const nl = buf.indexOf('\n'); // Python worker sends one line per result
      if (nl !== -1) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        worker.stdout.off('data', onData);
        try {
          const json = JSON.parse(line); // Parse the JSON result
          resolve(json);
        } catch (e) {
          reject(e);
        }
      }
    };

    // Attach listener and send JSON request
    worker.stdout.on('data', onData);
    worker.stdin.write(JSON.stringify({ image: imagePath, topk }) + '\n');
  });
}

// warm up the worker on boot (loads model once immediately)
(async () => {
  try {
    getWorker();
    console.log('[pyworker] started');
  } catch (e) {
    console.error('[pyworker] start error', e);
  }
})();

app.use(express.json()); // for parsing application/json
app.use(cors());  // Enable CORS so the mobile app can call this API

// serve uploaded images for frontend display
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
app.use('/uploads', express.static(UPLOAD_DIR));

// Multer storage for /upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR), // Save to uploads folder
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname || '') || '.jpg'; // Keep file extension
    cb(null, uuidv4() + ext.toLowerCase()); // Generate unique filename
  }
});
const upload = multer({ storage }); // Create Multer instance

// Health endpoint
app.get('/health', (req, res) => res.json({ ok: true }));

// Main endpoint, which uploads image, runs AI inference, stores results
app.post('/scan', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) { // Validate that an image was uploaded
      return res.status(400).json({ error: 'Image is required (field name: image).' });
    }

    // Read optional metadata fields from the request body
    const {
      user_id = null,
      location_latitude = null,
      location_longitude = null,
      source = 'camera',
      location_name = '',
      notes = null,
    } = req.body || {};

    // safe numeric defaults (avoid NOT NULL errors)
    const lat = Number.isFinite(Number(location_latitude)) ? Number(location_latitude) : 0.0;
    const lon = Number.isFinite(Number(location_longitude)) ? Number(location_longitude) : 0.0;

    // Compute absolute and public paths for the uploaded image
    const imagePathAbs = req.file.path.replace(/\\/g, '/');
    const imagePathPublic = `/uploads/${req.file.filename}`;

    // Send image to persistent Python worker for inference
    const result = await inferWithWorker(imagePathAbs, 5);
    console.log('[pyworker JSON]', result); // Log Python inference output

    // Safely extract confidence and check for "unsure" flag
    let confidence = Number(result.confidence); 
    if (!Number.isFinite(confidence)) confidence = 0;
    const unsure = confidence < Number(process.env.UNSURE_THRESHOLD || 0.6);

    // Insert observation metadata into MySQL
    const observation_id = await insertObservation({
      user_id: user_id ? Number(user_id) : null,
      species_id: null, // Initially null, will be linked later
      photo_url: imagePathPublic, // Image URL served by Express
      location_latitude: lat,
      location_longitude: lon,
      location_name,
      source: source || 'camera',
      status: 'pending',
      notes,
    });

    // Prepare and insert AI Top-K classification results
    const topk = Array.isArray(result.topk)
      ? result.topk
      : [{ name: result.species_name, confidence }]; // Fallback if no top-k list

    const items = [];
    let rank = 1;
    for (const t of topk) {
      const sci = String(t.name || t.species_name || '').trim();
      if (!sci) continue; // Skip invalid names
      const sid = await getOrCreateSpeciesId(sci); // Ensure species exists in DB
      const c = Number(t.confidence); // Confidence score
      items.push({ species_id: sid, confidence: Number.isFinite(c) ? c : 0, rank });
      rank += 1;
    }

    await insertAiResults(observation_id, items); // Insert Top-K results into ai_results table

    // Return combined observation + AI results as JSON
    const detail = await getObservationWithResults(observation_id);
    return res.json({
      observation_id,
      primary: {
        species_name: result.species_name,
        confidence, // 0..1
        unsure,
        image_path: imagePathPublic,
      },
      results: detail?.results || [],
      created_at: detail?.observation?.created_at,
    });
  } catch (e) { // Handle unexpected server or inference errors 
    console.error(e); 
    return res.status(500).json({ error: 'Server error' });
  }
});

const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, () => console.log(`SmartPlant API on http://localhost:${PORT}`));
