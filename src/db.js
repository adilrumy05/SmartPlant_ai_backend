import mysql from 'mysql2/promise'; // Promise based for async/await support
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create a connection pool for efficient MySQL database access
const pool = mysql.createPool({
  host: process.env.DB_HOST || '127.0.0.1', 
  port: process.env.DB_PORT ? Number(process.env.DB_PORT) : 3307, // Database port 
  user: process.env.DB_USER || 'root', // Database username
  password: process.env.DB_PASSWORD || '', // Database password 
  database: process.env.DB_NAME || 'sarawak_plant_db', // Default database name
  waitForConnections: true, // Queue requests if all connections are busy
  connectionLimit: 10, // Max number of concurrent connections
  queueLimit: 0, // Unlimited queue size for pending requests
  decimalNumbers: true // Return DECIMAL/NUMERIC fields as numbers (not strings)
});

export async function insertSpecies({
  scientific_name,
  common_name = null,
  is_endangered = false,
  description = null,
  image_url = null,
}) {
  if (!scientific_name) {
    throw new Error('scientific_name is required');
  }

  const endangeredFlag = is_endangered ? 1 : 0;

  const [result] = await pool.query(
    `
      INSERT INTO species
        (scientific_name, common_name, is_endangered, description, image_url, created_at)
      VALUES (?, ?, ?, ?, ?, NOW())
    `,
    [scientific_name, common_name, endangeredFlag, description, image_url]
  );

  return result.insertId;
}

export async function attachObservationToSpecies({
  observation_id,
  species_id,
  status = 'verified',
}) {
  if (!observation_id || !species_id) {
    throw new Error('observation_id and species_id are required');
  }

  await pool.query(
    `
      UPDATE plant_observations
      SET species_id = ?, status = ?
      WHERE observation_id = ?
    `,
    [species_id, status, observation_id]
  );
}

// Retrieve species_id from database if it exists, otherwise, insert a new record and return the new ID
export async function getOrCreateSpeciesId(scientific_name) {
  const conn = await pool.getConnection(); // Get a dedicated connection from the pool
  try {
    // Attempt to find an existing species by its scientific name
    const [rows] = await conn.query(
      'SELECT species_id FROM species WHERE scientific_name = ? LIMIT 1',
      [scientific_name]
    );
    if (rows.length) return rows[0].species_id; // If a match is found, return the existing species_id

    // If not found, create a placeholder row
    const [ins] = await conn.query(
      'INSERT INTO species (scientific_name) VALUES (?)',
      [scientific_name]
    );
    return ins.insertId; // Return the ID of the newly inserted species
  } finally {
    conn.release(); // Always release the connection back to the pool
  }
}

// Add a new observation record into the 'plant_observations' table
export async function insertObservation({
  user_id = null,
  species_id = null,
  photo_url,
  location_latitude = null,
  location_longitude = null,
  location_name = null,
  source = 'camera',
  status = 'pending',
  notes = null,
}) {
  /// Ensure latitude and longitude are numeric and non-null
  const lat =
    Number.isFinite(Number(location_latitude)) ? Number(location_latitude) : 0.0;
  const lon =
    Number.isFinite(Number(location_longitude)) ? Number(location_longitude) : 0.0;
  // Provide safe defaults for optional fields
  const locName = (location_name ?? '') || ''; // empty string instead of NULL
  const src = (source ?? 'camera') || 'camera';
  const st = (status ?? 'pending') || 'pending';
  const nts = notes ?? null; // notes can be NULL if not provided

  // Insert a new observation record
  const [res] = await pool.query(
    `INSERT INTO plant_observations
     (user_id, species_id, photo_url, location_latitude, location_longitude, location_name, source, status, notes)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      user_id,
      species_id,
      photo_url,
      lat,
      lon,
      locName,
      src,
      st,
      nts,
    ]
  );

  return res.insertId; // Return the new observation_id for later linking with AI results
}

// Insert Top-K AI classification results for an observation
export async function insertAiResults(observation_id, items) {
  if (!items?.length) return; // Exit early if no results to insert

  const values = [];

  // Prepare a 2D array of tuples for bulk insert
  for (const it of items) {
    const score = Math.max(0, Math.min(1, Number(it.confidence))); // Clamp confidence between 0 and 1
    const rank = Number(it.rank); // Rank (1 = top prediction)
    const species_id = Number(it.species_id); // Foreign key to 'species' table
    values.push([observation_id, species_id, Number(score.toFixed(4)), rank]); // Each subarray represents one row for insertion
  }

  // Bulk insert all AI results at once 
  await pool.query(
    `INSERT INTO ai_results (observation_id, species_id, confidence_score, rank)
     VALUES ?`,
    [values]
  );
}

// Fetch an observation record and its associated AI results, joined with species details
export async function getObservationWithResults(observation_id) {
  // Query to get the observation details
  const [obsRows] = await pool.query(
    `SELECT observation_id, user_id, species_id, photo_url, location_latitude, location_longitude,
            location_name, source, status, created_at
     FROM plant_observations WHERE observation_id = ?`,
    [observation_id]
  );
  const observation = obsRows[0] || null; // Use first row (only one per ID)
  if (!observation) return null; // Return null if no record found

  // Query to get all AI results linked to this observation
  const [resRows] = await pool.query(
    `SELECT ar.ai_result_id, ar.observation_id, ar.confidence_score, ar.rank,
            s.species_id, s.scientific_name, s.common_name
     FROM ai_results ar
     LEFT JOIN species s ON s.species_id = ar.species_id
     WHERE ar.observation_id = ?
     ORDER BY ar.rank ASC`, // Order by rank so Top-1 comes first
    [observation_id]
  );

  // Return combined data: observation + AI predictions
  return { observation, results: resRows };
}

export async function listObservationsByStatus(
  statuses = ['pending'],
  limit = 20,
  offset = 0,
  opts = {}
) {
  const { autoFlagged = false, threshold = null } = opts;

  const placeholders = statuses.map(() => '?').join(',');
  const topExpr = 'COALESCE(MAX(ar.confidence_score), 0)';

  const sqlParts = [];
  sqlParts.push(`
    SELECT
      po.observation_id,
      po.photo_url,
      po.status,
      ${topExpr} AS top_confidence,
      po.created_at,
      po.location_name,
      po.location_latitude,
      po.location_longitude,
      po.user_id,
      (
        SELECT s.scientific_name
        FROM ai_results ar2
        LEFT JOIN species s ON s.species_id = ar2.species_id
        WHERE ar2.observation_id = po.observation_id
        ORDER BY ar2.rank ASC, ar2.confidence_score DESC
        LIMIT 1
      ) AS top_species_name
    FROM plant_observations po
    LEFT JOIN ai_results ar ON ar.observation_id = po.observation_id
    WHERE po.status IN (${placeholders})
    GROUP BY
      po.observation_id,
      po.photo_url,
      po.status,
      po.created_at,
      po.location_name,
      po.location_latitude,
      po.location_longitude,
      po.user_id
  `);

  const params = [...statuses];

  if (autoFlagged && Number.isFinite(Number(threshold))) {
    sqlParts.push(`HAVING ${topExpr} < ?`);
    params.push(Number(threshold));
  }

  sqlParts.push(`ORDER BY po.created_at DESC LIMIT ? OFFSET ?`);
  params.push(Number(limit), Number(offset));

  const sql = sqlParts.join('\n');

  const [rows] = await pool.query(sql, params);
  return rows;
}

export async function updateObservationStatus(observation_id, status) {
  await pool.query(
    'UPDATE plant_observations SET status = ? WHERE observation_id = ?',
    [status, observation_id]
  );
}

export async function updateObservation({ observation_id, status, notes = null, species_name = null }) {
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();

    let species_id = null;
    if (species_name && String(species_name).trim()) {
      const [rows] = await conn.query('SELECT species_id FROM species WHERE scientific_name = ? LIMIT 1', [species_name.trim()]);
      if (rows.length) species_id = rows[0].species_id;
      else {
        const [ins] = await conn.query('INSERT INTO species (scientific_name) VALUES (?)', [species_name.trim()]);
        species_id = ins.insertId;
      }
    }

    const fields = [];
    const params = [];

    if (status) { fields.push('status = ?'); params.push(status); }
    if (notes !== undefined) { fields.push('notes = ?'); params.push(notes); }
    if (species_id !== null) { fields.push('species_id = ?'); params.push(species_id); }

    if (fields.length) {
      params.push(observation_id);
      await conn.query(`UPDATE plant_observations SET ${fields.join(', ')} WHERE observation_id = ?`, params);
    }

    await conn.commit();
  } catch (e) {
    await conn.rollback();
    throw e;
  } finally {
    conn.release();
  }
}

export default pool; // Export the pool instance (used in server.js for general queries)
