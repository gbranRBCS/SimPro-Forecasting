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

function toNum(x) {
  if (x === null || x === undefined) return null;
  const n = typeof x === 'number' ? x : parseFloat(String(x).replace(/[,£$]/g,''));
  return Number.isFinite(n) ? n : null;
}

function deriveRow(j) {
  // revenue fallback: revenue || Total.IncTax
  const rev = toNum(j.revenue ?? j.Total?.IncTax);
  // costs: prefer cost_total; else sum materials/labour/overhead
  const mat = toNum(j.materials) ?? 0;
  const lab = toNum(j.labour) ?? 0;
  const ovh = toNum(j.overhead) ?? 0;
  const cost_total = toNum(j.cost_total) ?? (mat + lab + ovh || null);

  // label: try netMarginPct; else compute if possible; else keep class
  let netMarginPct = j.netMarginPct;
  if (netMarginPct == null && rev && cost_total != null && rev > 0) {
    netMarginPct = (rev - cost_total) / rev;
  }

  // if still no margin, accept profitability_class 
  const profitability_class = j.profitability_class ?? null;

  // keep only if we have margin || class
  if (netMarginPct == null && !profitability_class) return null;

  return {
    ID: j.ID ?? j.id ?? null,
    revenue: rev ?? null,
    materials: toNum(j.materials),
    labour: toNum(j.labour),
    overhead: toNum(j.overhead),
    cost_total,
    statusName: j.statusName ?? null,
    jobType: j.jobType ?? null,
    customerName: j.customerName ?? null,
    siteName: j.siteName ?? null,
    job_age_days: j.job_age_days ?? null,
    lead_time_days: j.lead_time_days ?? null,
    is_overdue: j.is_overdue ?? 0,
    descriptionText: j.descriptionText ?? "",
    ...(netMarginPct != null ? { netMarginPct } : {}),
    ...(profitability_class ? { profitability_class } : {})
  };
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

  await axios.get(`${BASE}/data/sync`, { headers: h });

  const jobsRes = await axios.get(`${BASE}/data/jobs`, {
    headers: h,
    params: { date: '2019-01-01,2035-12-31', order: 'date', limit:50 }
  });

  const jobs = jobsRes.data?.jobs ?? jobsRes.data ?? [];
  const rows = [];
  for (const j of jobs) {
    const r = deriveRow(j);
    if (r) rows.push(r);
  }

  const payload = {
    data: rows,
    test_size: 0.2,
    random_state: 42,
    max_tfidf_features: 500
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