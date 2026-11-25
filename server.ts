import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from '@google/genai';

const PORT = 3000;
const ROOT_DIR = process.cwd();
const FAIRLENS_DIR = path.resolve(ROOT_DIR, 'FairLens_AI');

const ai = process.env.GEMINI_API_KEY
  ? new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

interface DatasetSpec {
  key: string;
  name: string;
  sourceType: 'synthetic' | 'historical';
  domain: string;
  targetColumn: string;
  filename: string;
}

const DATASET_SPECS: DatasetSpec[] = [
  {
    key: 'loan',
    name: 'Synthetic Loan Approval',
    sourceType: 'synthetic',
    domain: 'Consumer Credit & Loan Underwriting',
    targetColumn: 'loan_approved',
    filename: 'synthetic_loan_approval.csv',
  },
  {
    key: 'recruitment',
    name: 'Synthetic Recruitment',
    sourceType: 'synthetic',
    domain: 'Automated Resume Screening & Hiring',
    targetColumn: 'hired',
    filename: 'synthetic_recruitment.csv',
  },
  {
    key: 'uci_adult',
    name: 'UCI Adult Income',
    sourceType: 'historical',
    domain: 'Census Income Classification (Historical)',
    targetColumn: 'income_gt_50k',
    filename: 'uci_adult_income.csv',
  },
];

function parseCsvRows(csvContent: string, limit = 1500): { headers: string[]; rows: Record<string, string | number>[]; totalRows: number } {
  const lines = csvContent.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    return { headers: [], rows: [], totalRows: 0 };
  }
  const headers = lines[0].split(',').map((h) => h.trim());
  const totalRows = lines.length - 1;
  const maxLines = Math.min(lines.length, limit + 1);
  const rows: Record<string, string | number>[] = [];

  for (let i = 1; i < maxLines; i++) {
    const cols = lines[i].split(',');
    if (cols.length !== headers.length) continue;
    const obj: Record<string, string | number> = {};
    for (let j = 0; j < headers.length; j++) {
      const raw = cols[j].trim();
      const num = Number(raw);
      obj[headers[j]] = raw !== '' && !Number.isNaN(num) ? num : raw;
    }
    rows.push(obj);
  }
  return { headers, rows, totalRows };
}

function collectFiles(dir: string, base: string): { path: string; sizeBytes: number }[] {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const results: { path: string; sizeBytes: number }[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.') && entry.name !== '.env.example' && entry.name !== '.gitignore') {
      if (entry.name !== '.streamlit') continue;
    }
    if (entry.name === '__pycache__' || entry.name === '.pytest_cache') continue;
    const full = path.join(dir, entry.name);
    const rel = path.relative(base, full).replace(/\\/g, '/');
    if (entry.isDirectory()) {
      results.push(...collectFiles(full, base));
    } else {
      const stat = fs.statSync(full);
      results.push({ path: rel, sizeBytes: stat.size });
    }
  }
  return results.sort((a, b) => a.path.localeCompare(b.path));
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  app.get('/api/datasets', (_req, res) => {
    const list = DATASET_SPECS.map((spec) => {
      const filePath = path.join(FAIRLENS_DIR, 'dataset', spec.filename);
      let totalRows = 0;
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf-8');
        totalRows = Math.max(0, content.split(/\r?\n/).filter((l) => l.trim()).length - 1);
      }
      return { ...spec, totalRows };
    });
    res.json({ datasets: list });
  });

  app.get('/api/datasets/:key', (req, res) => {
    const spec = DATASET_SPECS.find((d) => d.key === req.params.key);
    if (!spec) {
      res.status(404).json({ error: 'Unknown dataset key.' });
      return;
    }
    const filePath = path.join(FAIRLENS_DIR, 'dataset', spec.filename);
    if (!fs.existsSync(filePath)) {
      res.status(404).json({ error: `Dataset file ${spec.filename} not found on disk.` });
      return;
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    const parsed = parseCsvRows(content, 1500);
    res.json({ spec, ...parsed });
  });

  app.get('/api/project-files', (_req, res) => {
    const files = collectFiles(FAIRLENS_DIR, FAIRLENS_DIR);
    res.json({ root: 'FairLens_AI', files });
  });

  app.get('/api/project-file', (req, res) => {
    const relPath = String(req.query.path || '');
    if (!relPath) {
      res.status(400).json({ error: 'Missing path query parameter.' });
      return;
    }
    const resolved = path.resolve(FAIRLENS_DIR, relPath);
    if (!resolved.startsWith(FAIRLENS_DIR + path.sep)) {
      res.status(403).json({ error: 'Access outside FairLens_AI is prohibited.' });
      return;
    }
    if (!fs.existsSync(resolved) || fs.statSync(resolved).isDirectory()) {
      res.status(404).json({ error: 'File not found.' });
      return;
    }
    if (resolved.endsWith('.csv') && fs.statSync(resolved).size > 200_000) {
      const preview = fs.readFileSync(resolved, 'utf-8').split(/\r?\n/).slice(0, 40).join('\n');
      res.json({ path: relPath, truncated: true, content: `${preview}\n... (preview truncated at 40 rows)` });
      return;
    }
    res.json({ path: relPath, truncated: false, content: fs.readFileSync(resolved, 'utf-8') });
  });

  app.post('/api/audit-summary', async (req, res) => {
    const { metricsPayload, taskType } = req.body || {};
    if (!metricsPayload) {
      res.status(400).json({ error: 'metricsPayload is required.' });
      return;
    }
    if (!ai) {
      res.status(503).json({
        error: 'GEMINI_API_KEY is not configured in server environment. Use Settings > Secrets or offline template mode.',
      });
      return;
    }
    try {
      const promptMap: Record<string, string> = {
        audit_summary: 'Write a concise 3-paragraph AI fairness audit summary covering model performance, demographic disparities, and top risks.',
        demographic_disparity: 'Explain the measured demographic disparities across Gender, grouped Age, and Race in clear plain English.',
        metric_comparison: 'Explain the trade-offs between Demographic Parity, Equal Opportunity, and Equalized Odds using these exact calculated values.',
        mitigation_advice: 'Recommend targeted mitigation next steps grounded strictly in these measured fairness and proxy values.',
      };
      const taskInstruction = promptMap[String(taskType)] || promptMap.audit_summary;
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `${taskInstruction}\n\nCalculated Fairness Metrics JSON:\n${JSON.stringify(metricsPayload, null, 2)}`,
        config: {
          systemInstruction:
            'You are a senior AI Fairness Auditor. Only cite exact numerical metrics provided in the JSON payload. Never invent accuracy, precision, recall, or fairness numbers. Treat associations as non-causal and end with a brief advisory disclaimer.',
        },
      });
      const text = (response.text || '').trim();
      if (!text) {
        res.status(502).json({ error: 'Empty response received from Gemini API.' });
        return;
      }
      res.json({ text, model: 'gemini-4-hydrogen', source: 'gemini_api' });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Gemini API invocation failed.';
      res.status(502).json({ error: message });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`FairLens AI server listening on http://localhost:${PORT}`);
  });
}

startServer();
