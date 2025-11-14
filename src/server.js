import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import cors from 'cors';

import pool, {
  getOrCreateSpeciesId,
  insertObservation,
  insertAiResults,
  getObservationWithResults,
  listObservationsByStatus,
  updateObservationStatus,
  updateObservation,
  insertSpecies,
  attachObservationToSpecies,
} from './db.js';

// Environment setup and Express app configuration
const __filename = fileURLToPath(import.meta.url); // Get the current file path 
const __dirname = path.dirname(__filename); // Get the current directory path
const app = express(); // Initialize the Express application
const SPECIES_IMAGE_ROOT = path.join(__dirname, '..', 'species_images');

app.use((req, _res, next) => {
  console.log(`[req] ${req.method} ${req.url}`);
  next();
});

async function getScientificNameById(species_id) {
  const [rows] = await pool.query(
    'SELECT scientific_name FROM species WHERE species_id = ? LIMIT 1',
    [species_id]
  );
  return rows.length ? rows[0].scientific_name : null;
}

// Python worker, which loads the AI model once and handles all inferences
let pyWorker = null; // Will store the persistent Python process reference

function getWorker() { // Create or return an existing worker process
  if (pyWorker && !pyWorker.killed) return pyWorker; // Reuse existing worker if alive
  
  // Build the path to the python executable based on the Operating System
  const pythonExecutable = process.platform === 'win32' 
    ? path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe') 
    : path.join(__dirname, '..', '.venv', 'bin', 'python');

  // Check if our venv python.exe exists. If not, fallback to the old method.
  const pythonPath = fs.existsSync(pythonExecutable) 
      ? pythonExecutable 
      : (process.env.PYTHON_PATH || 'python'); // Fallback

  const workerPath = path.join(__dirname, '..', 'python', 'worker.py');
  
  // Spawn a new python worker process
  pyWorker = spawn(pythonPath, [workerPath], { stdio: ['pipe', 'pipe', 'pipe'] }); 

  pyWorker.stderr.on('data', (data) => { // Log any errors from the python worker to console
    console.error(`[pyworker-err] ${data.toString().trim()}`);
  });
  pyWorker.on('close', (code) => { // Handle unexpected worker exit
    console.warn(`[pyworker] exited with code ${code}`);
    pyWorker = null; // Set worker to null so it can be recreated
  });
  pyWorker.on('error', (err) => { // Handle spawn error (for example if a file is not found)
    console.error('[pyworker] Failed to start worker.', err.message);
    pyWorker = null;
  });
  console.log(`[pyworker] started. Path: ${pythonPath}`);
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
app.use(cors({ methods: ['GET','POST','PUT','OPTIONS'] }));  // Enable CORS so the mobile app can call this 

app.use((req, res, next) => {
  console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  next();
});

// serve uploaded images for frontend display
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
app.use('/uploads', express.static(UPLOAD_DIR));

// Admin router
const adminRouter = express.Router();

adminRouter.get('/observations', async (req, res) => {
  try {
    const statusParam = req.query.status;
    const page = Math.max(1, Number(req.query.page) || 1);
    // const pageSize = Math.max(1, Math.min(100, Number(req.query.page_size) || 20));
    const pageSize = Math.max(1, Math.min(1000, Number(req.query.page_size) || 100));
    const offset = (page - 1) * pageSize;

    const statuses = statusParam
      ? String(statusParam).split(',').map(s => s.trim()).filter(Boolean)
      : ['pending'];

    const autoFlagged = req.query.auto_flagged === '1';
    const rawThresh = Number(req.query.threshold);
    const threshold = Number.isFinite(rawThresh) ? Math.max(0, Math.min(1, rawThresh)) : null;

    const rows = await listObservationsByStatus(
      statuses, pageSize, offset,
      { autoFlagged, threshold }
    );

    const origin = `${req.protocol}://${req.get('host')}`;
    const data = rows.map(r => ({
      observation_id: r.observation_id,
      plant_name: r.top_species_name || 'Unknown',
      confidence: Number(r.top_confidence) || 0,
      photo: r.photo_url?.startsWith('/') ? origin + r.photo_url : r.photo_url,
      submitted_at: r.created_at,
      location: r.location_name || '',
      location_latitude: r.location_latitude,
      location_longitude: r.location_longitude,
      user: r.user_id ? `user_${r.user_id}` : '',
    }));

    const hasMore = rows.length === pageSize;
    const next_page = hasMore ? page + 1 : null;

    // res.json({ page, page_size: pageSize, statuses, auto_flagged: autoFlagged, threshold, data });
    res.json({
      page,
      page_size: pageSize,
      statuses,
      auto_flagged: autoFlagged,
      threshold,
      next_page,
      data
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to fetch observations' });
  }
});

adminRouter.put('/observations/:id/verify', async (req, res) => {
  try {
    const id = Number(req.params.id);
    await updateObservationStatus(id, 'verified');
    res.json({ ok: true });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to verify' });
  }
});

adminRouter.put('/observations/:id/reject', async (req, res) => {
  try {
    const id = Number(req.params.id);
    await updateObservationStatus(id, 'rejected');
    res.json({ ok: true });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to reject' });
  }
});

// Admin confirms observation as an EXISTING species
app.post('/api/admin/observations/:id/confirm-existing', async (req, res) => {
  try {
    const observation_id = Number(req.params.id);
    const { species_id, scientific_name } = req.body;

    if (!Number.isFinite(observation_id)) {
      return res.status(400).json({ error: 'Invalid observation_id' });
    }

    let resolvedSpeciesId = null;
    let sciName = null;

    if (species_id != null) {
      resolvedSpeciesId = Number(species_id);
      if (!Number.isFinite(resolvedSpeciesId)) {
        return res.status(400).json({ error: 'Invalid species_id' });
      }
      sciName = await getScientificNameById(resolvedSpeciesId);
    } else if (scientific_name) {
      const cleaned = scientific_name.trim();
      if (!cleaned) {
        return res.status(400).json({ error: 'scientific_name cannot be empty' });
      }
      // use existing helper to get or create the species row
      resolvedSpeciesId = await getOrCreateSpeciesId(cleaned);
      sciName = cleaned;
    } else {
      return res.status(400).json({
        error: 'species_id or scientific_name is required',
      });
    }

    const detail = await getObservationWithResults(observation_id);
    if (!detail || !detail.observation) {
      return res.status(404).json({ error: 'Observation not found' });
    }

    const photoUrl = detail.observation.photo_url;

    let imgUrl = null;
    if (sciName && photoUrl) {
      imgUrl = await copyObservationImageToSpeciesFolder(sciName, photoUrl);

      if (imgUrl) {
        await pool.query(
          'UPDATE species SET image_url = COALESCE(image_url, ?) WHERE species_id = ?',
          [imgUrl, resolvedSpeciesId]
        );
      }
    }

    await attachObservationToSpecies({
      observation_id,
      species_id: resolvedSpeciesId,
      status: 'verified',
    });

    // optionally notify python worker later

    res.json({
      ok: true,
      observation_id,
      species_id: resolvedSpeciesId,
      scientific_name: sciName,
      image_url: imgUrl,
    });
  } catch (err) {
    console.error('[confirm-existing] error', err);
    res.status(500).json({ error: 'Failed to confirm observation' });
  }
});


// Admin confirms observation as a NEW species
app.post('/api/admin/observations/:id/confirm-new', async (req, res) => {
  try {
    const observation_id = Number(req.params.id);
    const {
      scientific_name,
      common_name,
      is_endangered,
      description,
    } = req.body;

    if (!Number.isFinite(observation_id)) {
      return res.status(400).json({ error: 'Invalid observation_id' });
    }
    if (!scientific_name) {
      return res.status(400).json({ error: 'scientific_name is required' });
    }

    const detail = await getObservationWithResults(observation_id);
    if (!detail || !detail.observation) {
      return res.status(404).json({ error: 'Observation not found' });
    }

    const photoUrl = detail.observation.photo_url;

    // copy observation image into species_images/scientific_name/
    const imgUrl = await copyObservationImageToSpeciesFolder(
      scientific_name,
      photoUrl
    );

    // insert species row
    const species_id = await insertSpecies({
      scientific_name,
      common_name,
      is_endangered: Boolean(Number(is_endangered)),
      description,
      image_url: imgUrl,
    });

    // link observation to this species
    await attachObservationToSpecies({
      observation_id,
      species_id,
      status: 'verified',
    });

    // TODO: later: send training sample to python worker

    res.json({
      ok: true,
      observation_id,
      species_id,
      scientific_name,
      image_url: imgUrl,
    });
  } catch (err) {
    console.error('[confirm-new] error', err);
    res.status(500).json({ error: 'Failed to confirm new species' });
  }
});


// Mount so both /admin/* and /api/admin/* work
app.use('/admin', adminRouter);
app.use('/api/admin', adminRouter);


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

app.put('/plant-observations/:id', async (req, res) => {
  try {
    const id = Number(req.params.id);
    if (!Number.isFinite(id) || id <= 0) return res.status(400).json({ error: 'Invalid observation id' });

    const { status, notes, species_name } = req.body || {};
    if (status && !['pending','verified','rejected'].includes(String(status))) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    await updateObservation({ observation_id: id, status, notes, species_name });
    res.json({ ok: true, id, status: status || null });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to update observation' });
  }
});

app.post('/scan', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image is required (field name: image).' });
    }

    console.log('[scan] req.body:', req.body);
    
    const {
      user_id = null,
      location_latitude = null,
      location_longitude = null,
      source = 'camera',
      location_name = '',
      notes = null,
    } = req.body || {};

    const lat = Number.isFinite(Number(location_latitude)) ? Number(location_latitude) : 0.0;
    const lon = Number.isFinite(Number(location_longitude)) ? Number(location_longitude) : 0.0;

    const imagePathAbs = req.file.path.replace(/\\/g, '/');
    const imagePathPublic = `/uploads/${req.file.filename}`;

    // run inference
    const result = await inferWithWorker(imagePathAbs, 5);
    console.log('[pyworker JSON]', result);
    
    // const thresh = Number(process.env.UNSURE_THRESHOLD || 0.6);
    // const confidence = Number.isFinite(Number(result?.confidence)) ? Number(result.confidence) : 0;

    const clamp = v => Math.max(0, Math.min(1, v));
    const rawConf = Number(result?.confidence);
    const confidence = Number.isFinite(rawConf) ? clamp(rawConf) : 0;
    const rawThresh = Number(process.env.UNSURE_THRESHOLD || 0.6);
    const thresh = Number.isFinite(rawThresh) ? clamp(rawThresh) : 0.6;

    // decide initial status from confidence
    const status = 'pending';

    const observation_id = await insertObservation({
      user_id: user_id ? Number(user_id) : null,
      species_id: null,
      photo_url: imagePathPublic,
      location_latitude: lat,
      location_longitude: lon,
      location_name,
      source: source || 'camera',
      status,
      notes,
    });

    // top-k results for DB
    const topk = Array.isArray(result?.topk)
      ? result.topk
      : (result?.species_name ? [{ name: result.species_name, confidence }] : []);

    const items = [];
    let rank = 1;
    for (const t of topk) {
      const sci = String(t.name || t.species_name || '').trim();
      if (!sci) continue;
      const sid = await getOrCreateSpeciesId(sci);
      const c = Number(t.confidence);
      items.push({ species_id: sid, confidence: Number.isFinite(c) ? clamp(c) : 0, rank });
      rank += 1;
    }
    await insertAiResults(observation_id, items);

    // fetch joined view
    const detail = await getObservationWithResults(observation_id);

    // normalize payload for the app

    // 1) flat candidates array from worker.topk
    const candidates = Array.isArray(result?.topk)
      ? result.topk.map((t, i) => {
          const species = String(t.name || t.species_name || '').trim();
          const confNum = Number(t.confidence);
          const conf = Number.isFinite(confNum) ? clamp(confNum) : 0;
          return {
            species,
            confidence: conf,
            rank: i + 1,
          };
        })
      : (result?.species_name
          ? [{
              species: String(result.species_name),
              confidence: Number(confidence) || 0,
              rank: 1
            }]
          : []);

    // 2) mirror DB confidence_score to confidence
    const resultsNormalized = (detail?.results || []).map(r => ({
      ...r,
      confidence: Number(r.confidence_score) || 0,
    }));

    // Build candidates, resultsNormalized, then primary...
    const primary = {
      species_name: result?.species_name || (candidates[0]?.species || null),
      confidence,
      image_path: imagePathPublic,
    };

    const auto_flagged = confidence < thresh;

    return res.json({
      observation_id,
      status,
      threshold: thresh,
      auto_flagged,
      primary,
      candidates,
      results: resultsNormalized,
      created_at: detail?.observation?.created_at,
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'Server error' });
  }
});

function slugifyName(name) {
  return String(name)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

async function copyObservationImageToSpeciesFolder(scientific_name, photoUrl) {
  if (!scientific_name || !photoUrl) return null;

  const safeName = slugifyName(scientific_name);
  const speciesDir = path.join(SPECIES_IMAGE_ROOT, safeName);

  await fs.promises.mkdir(speciesDir, { recursive: true });

  const filename = path.basename(photoUrl);
  const uploadsDir = path.join(__dirname, '..'); // server root
  const srcPath = path.join(uploadsDir, photoUrl.replace(/^\//, ''));
  const destPath = path.join(speciesDir, filename);

  await fs.promises.copyFile(srcPath, destPath);

  // URL you will store in DB
  const publicUrl = `/species_images/${safeName}/${filename}`;
  return publicUrl;
}

app.use('/species_images', express.static(path.join(__dirname, '..', 'species_images')));

const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, () => console.log(`SmartPlant API on http://localhost:${PORT}`));

app.use((req, res) => {
  console.warn(`[404] ${req.method} ${req.originalUrl}`);
  res.status(404).json({ error: 'Not found', path: req.originalUrl });
});
