/**
SimPRO Data Build Tool
----------------------
Fetches historical job data from the backend API, computes metrics (profit margins, completion times),
and prepares a clean JSON dataset for training the ML models.
Usage:
  node ml/tools/build-train.js --user <user> --pass <pass> [options]
  node ml/tools/build-train.js --token <features> [options]

Options:
 --out <path>    Output file path (default: ml/train.json)
 --limit <num>   Max jobs to fetch (default: 500)
 --from <date>   Sync start date (default: 2015-01-01)
 --to <date>     Sync end date (default: today)
*/

import fs from 'fs';
import path from 'path';
import process from 'process';
import axios from 'axios';

// --- Configuration & Helpers ---

function arg(name, def = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : def;
}

// Default to a reasonably current date if not specified
const TODAY = new Date().toISOString().split('T')[0];

const CONFIG = {
  base: arg('base', 'http://localhost:5001'),
  user: arg('user'),
  pass: arg('pass'),
  token: arg('token'),
  out: arg('out', 'ml/train.json'),
  
  // Thresholds for profitability classification
  absLow: Number(arg('low', '0.44')),
  absHigh: Number(arg('high', '0.64')),

  // Sync settings
  from: arg('from', '2015-01-01'),
  to: arg('to', TODAY),
  limit: Number(arg('limit', '500')),
};

function toNum(x) {
  if (x === null || x === undefined) return null;
  const n = typeof x === 'number' ? x : parseFloat(String(x).replace(/[,£$]/g, ''));
  return Number.isFinite(n) ? n : null;
}

/**
Classifies net margin into Low/Medium/High buckets.
 */
function computeProfitClassFromMargin(p, low, high) {
  if (p == null) return null;
  if (p > high) return "High";
  if (p >= low) return "Medium";
  return "Low";
}

/**
Transforms a raw SimPRO job object into a flat feature definition for the ML model.
 */
function deriveRow(j) {
  // 1. Extract Financials
  // Revenue fallback: revenue || Total.IncTax
  const rev = toNum(j.revenue ?? j.Total?.IncTax);

  // Costs: prefer *_cost_est and cost_est_total; else fall back to legacy keys
  const mat = toNum(j.materials_cost_est ?? j.materials) ?? 0;
  const lab = toNum(j.labor_cost_est ?? j.labour) ?? 0;
  const ovh = toNum(j.overhead_est ?? j.overhead) ?? 0;
  
  // Total Cost
  const cost_total = toNum(j.cost_est_total ?? j.cost_total) ?? (mat + lab + ovh || null);

  // 2. Compute Margins
  // Try pre-calculated netMarginPct; else compute if possible.
  let netMarginPct = j.netMarginPct;
  if (netMarginPct == null && rev && cost_total != null && rev > 0) {
    netMarginPct = (rev - cost_total) / rev;
  }

  // If we can't determine a margin or existing class, discard row
  if (netMarginPct == null && !j.profitability_class) return null;

  return {
    ID: j.ID ?? j.id ?? null,
    
    // Financial features
    revenue: rev ?? null,
    materials: mat,
    labour: lab,
    overhead: ovh,
    cost_total,
    
    // Categorical features
    statusName: j.statusName ?? j.status_name ?? null,
    jobType: j.jobType ?? null,
    customerName: j.customerName ?? null,
    siteName: j.siteName ?? null,
    
    // Date features
    dateIssued: j.dateIssued ?? null,
    dateDue: j.dateDue ?? null,
    dateCompleted: j.dateCompleted ?? null,
    
    // Calculated time features
    job_age_days: j.age_days ?? j.job_age_days ?? null,
    lead_time_days: j.due_in_days ?? j.lead_time_days ?? null,
    completion_days: j.completion_days ?? null,  // Target variable for Duration Model
    
    // Status flags
    is_completed: j.is_completed ?? false,
    is_overdue: j.is_overdue ?? 0,

    // Text features (NLP)
    descriptionText: j.descriptionText ?? "",
    desc_len: j.desc_len ?? (j.descriptionText?.length ?? 0),
    has_emergency: j.has_emergency ?? 0,

    // Target variables
    ...(netMarginPct != null ? { netMarginPct } : {}),
    ...(j.profitability_class ? { profitability_class: j.profitability_class } : {})
  };
}

/**
Calculates a quantile value from an array of numbers.
q=0.5 is median, q=0.25 is lower quartile, etc.
*/
function quantile(arr, q) {
  const sorted = arr.slice().sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

// --- Main Execution ---

async function main() {
  // 1. Authentication
  let jwt = CONFIG.token;
  if (!jwt) {
    if (!CONFIG.user || !CONFIG.pass) {
      console.error('Error: Provide --token OR (--user and --pass).');
      process.exit(1);
    }
    try {
      console.log(`Authenticating as ${CONFIG.user}...`);
      const auth = await axios.post(`${CONFIG.base}/auth/login`, { 
        username: CONFIG.user, 
        password: CONFIG.pass 
      });
      jwt = auth.data.token;
    } catch (err) {
      console.error('Login failed:', err.response?.data?.message || err.message);
      process.exit(1);
    }
  }

  const headers = { Authorization: `Bearer ${jwt}` };

  // 2. Trigger Backup Sync
  console.log(`\n--- Syncing Data ---`);
  console.log(`Range: ${CONFIG.from} to ${CONFIG.to}`);
  try {
    await axios.get(`${CONFIG.base}/data/sync`, { 
      headers, 
      params: { 
        DateIssuedFrom: CONFIG.from, 
        DateIssuedTo: CONFIG.to, 
        force: 1 
      } 
    });
    console.log('Sync initialized/completed successfully.');
  } catch (err) {
    console.warn('Warning: Data sync request failed. Proceeding with existing data...');
    console.warn(`Reason: ${err.message}`);
  }

  // 3. Fetch Jobs
  console.log(`\n--- Fetching Training Data ---`);
  console.log(`Limit: ${CONFIG.limit} jobs`);
  
  let jobs = [];
  try {
    const jobsRes = await axios.get(`${CONFIG.base}/data/jobs`, {
      headers,
      params: { order: 'date', limit: CONFIG.limit }
    });
    jobs = jobsRes.data?.jobs ?? jobsRes.data ?? [];
    console.log(`Fetched ${jobs.length} raw jobs from API.`);
  } catch (err) {
    console.error('Failed to fetch jobs:', err.message);
    process.exit(1);
  }

  // 4. Process & Transform
  const rows = [];
  const margins = [];
  
  for (const j of jobs) {
    const r = deriveRow(j);
    if (r) {
      rows.push(r);
      if (r.netMarginPct != null) {
        margins.push(r.netMarginPct);
      }
    }
  }
  console.log(`Processed ${rows.length} valid training rows.`);

  // 5. Dynamic Thresholds
  let low = CONFIG.absLow;
  let high = CONFIG.absHigh;
  
  if (margins.length > 50) { // Require decent sample size for auto-thresholds
    low = quantile(margins, 0.33);
    high = quantile(margins, 0.67);
    console.log(`\nComputed dynamic thresholds (n=${margins.length}):`);
    console.log(`  Low (< 33%):     ${(low * 100).toFixed(1)}% margin`);
    console.log(`  High (> 67%):    ${(high * 100).toFixed(1)}% margin`);
  } else {
    console.log(`\nUsing default thresholds (n=${margins.length} insufficient for dynamic):`);
    console.log(`  Low: ${low}, High: ${high}`);
  }

  // 6. Label Generation
  const counts = { Low: 0, Medium: 0, High: 0 };
  for (const r of rows) {
    if (r.netMarginPct != null) {
      const cls = computeProfitClassFromMargin(r.netMarginPct, low, high);
      r.profitability_class = cls;
      counts[cls] = (counts[cls] || 0) + 1;
    } else if (r.profitability_class) {
      counts[r.profitability_class] = (counts[r.profitability_class] || 0) + 1;
    }
  }

  console.log('Class Distribution:', JSON.stringify(counts));

  // 7. Output Result
  const payload = {
    meta: {
      generatedAt: new Date().toISOString(),
      source: "build-train.js",
      recordCount: rows.length
    },
    data: rows,
    test_size: 0.2,
    random_state: 42,
    max_tfidf_features: 500,
    thresholds: { low, high }
  };

  try {
    fs.mkdirSync(path.dirname(CONFIG.out), { recursive: true });
    fs.writeFileSync(CONFIG.out, JSON.stringify(payload, null, 2));
    
    console.log(`\n--- Success ---`);
    console.log(`Training dataset written to: ${CONFIG.out}`);
    
    const haveMargin = rows.filter(r => r.netMarginPct != null).length;
    const haveClass  = rows.filter(r => r.profitability_class).length;
    console.log(`Metrics: ${haveMargin} rows with exact margin, ${haveClass} rows with class label.`);
    
  } catch (err) {
    console.error('Failed to write output file:', err.message);
    process.exit(1);
  }
}

main().catch(e => {
  console.error('Fatal Error defined:', e?.response?.data ?? e.message);
  process.exit(1);
});