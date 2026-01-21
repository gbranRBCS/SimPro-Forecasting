import fs from 'fs';
import path from 'path';
import process from 'process';
import axios from 'axios';

function arg(name, def=null) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i+1] : def;
}

const BASE = arg('base', 'http://localhost:5001');
const USER = arg('user');
const PASS = arg('pass');
const TOKEN = arg('token');
const OUT  = arg('out', 'ml/train.json');
const ABS_LOW = Number(arg('low', '0.44'));
const ABS_HIGH = Number(arg('high', '0.64'));

// date-range + limit arguments for historical data sync
const FROM = arg('from', '2015-01-01');           // default start (very old)
const TO   = arg('to', '2024-03-31');             // default end: end of 2024 Q1
const LIMIT = Number(arg('limit', '500'));         // how many jobs to read after syncing

function toNum(x) {
  if (x === null || x === undefined) return null;
  const n = typeof x === 'number' ? x : parseFloat(String(x).replace(/[,£$]/g,''));
  return Number.isFinite(n) ? n : null;
}

function computeProfitClassFromMargin(p, low, high) {
  if (p == null) return null;
  if (p > high) return "High";
  if (p >= low) return "Medium";
  return "Low";
}

function deriveRow(j) {
  // revenue fallback: revenue || Total.IncTax
  const rev = toNum(j.revenue ?? j.Total?.IncTax);

  // costs: prefer *_cost_est and cost_est_total; else fall back to legacy keys
  const mat = toNum(j.materials_cost_est ?? j.materials) ?? 0;
  const lab = toNum(j.labor_cost_est ?? j.labour) ?? 0;
  const ovh = toNum(j.overhead_est ?? j.overhead) ?? 0;
  const cost_total = toNum(j.cost_est_total ?? j.cost_total) ?? (mat + lab + ovh || null);

  // label: try netMarginPct; else compute if possible; else keep class
  let netMarginPct = j.netMarginPct;
  if (netMarginPct == null && rev && cost_total != null && rev > 0) {
    netMarginPct = (rev - cost_total) / rev;
  }


  
  if (netMarginPct == null && !j.profitability_class) return null;

  return {
    ID: j.ID ?? j.id ?? null,
    revenue: rev ?? null,
    materials: mat,
    labour: lab,
    overhead: ovh,
    cost_total,
    statusName: j.statusName ?? j.status_name ?? null,
    jobType: j.jobType ?? null,
    customerName: j.customerName ?? null,
    siteName: j.siteName ?? null,
    
    // date features
    dateIssued: j.dateIssued ?? null,
    dateDue: j.dateDue ?? null,
    dateCompleted: j.dateCompleted ?? null,
    
    // calculated time features
    job_age_days: j.age_days ?? j.job_age_days ?? null,
    lead_time_days: j.due_in_days ?? j.lead_time_days ?? null,
    completion_days: j.completion_days ?? null,  // NEW: actual completion time
    
    // status flags
    is_completed: j.is_completed ?? false,
    is_overdue: j.is_overdue ?? 0,

    // text features
    descriptionText: j.descriptionText ?? "",
    desc_len: j.desc_len ?? (j.descriptionText?.length ?? 0),
    has_emergency: j.has_emergency ?? 0,

    // target variables
    ...(netMarginPct != null ? { netMarginPct } : {}),
    ...(j.profitability_class ? { profitability_class: j.profitability_class } : {})
  };
}

function quantile(arr, q) {
  const sorted = arr.slice().sort((a,b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

async function main() {
  let jwt = TOKEN;
  if (!jwt) {
    if (!USER || !PASS) {
      console.error('Provide --token OR (--user and --pass).');
      process.exit(1);
    }
    const auth = await axios.post(`${BASE}/auth/login`, { username: USER, password: PASS });
    jwt = auth.data.token;
  }
  const h = { Authorization: `Bearer ${jwt}` };

  // ask backend to sync historical range
  console.log(`Requesting sync for DateIssuedFrom=${FROM} DateIssuedTo=${TO}`);
  await axios.get(`${BASE}/data/sync`, { headers: h, params: { DateIssuedFrom: FROM, DateIssuedTo: TO, force: 1 } });


  const jobsRes = await axios.get(`${BASE}/data/jobs`, {
    headers: h,
    params: { order: 'date', limit: LIMIT }
  });

  const jobs = jobsRes.data?.jobs ?? jobsRes.data ?? [];
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

  // calcluatte dynamic thresholds if enough data, else default to predetermined
  let low = ABS_LOW;
  let high = ABS_HIGH;
  
  if (margins.length > 10) {
      low = quantile(margins, 0.33);
      high = quantile(margins, 0.67);
      console.log(`Computed dynamic thresholds from ${margins.length} margins: Low=${low.toFixed(3)}, High=${high.toFixed(3)}`);
  } else {
      console.log(`Using default thresholds: Low=${low}, High=${high}`);
  }

  let counts = { Low: 0, Medium: 0, High: 0 };
  
  for (const r of rows) {
      if (r.netMarginPct != null) {
          const cls = computeProfitClassFromMargin(r.netMarginPct, low, high);
          r.profitability_class = cls;
          counts[cls] = (counts[cls] || 0) + 1;
      } else if (r.profitability_class) {
          counts[r.profitability_class] = (counts[r.profitability_class] || 0) + 1;
      }
  }

  console.log('Class distribution:', counts);

  const payload = {
    data: rows,
    test_size: 0.2,
    random_state: 42,
    max_tfidf_features: 500,
    thresholds: { low, high }
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 2));
  console.log(`Wrote ${rows.length} rows to ${OUT}`);
  const haveMargin = rows.filter(r => r.netMarginPct != null).length;
  const haveClass  = rows.filter(r => r.profitability_class).length;
  console.log({ haveMargin, haveClass });
}

main().catch(e => {
  console.error('build-train failed:', e?.response?.data ?? e);
  process.exit(1);
});