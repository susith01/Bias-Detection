export type AgeGroup = '≤25' | '26–35' | '36–45' | '46–55' | '56+' | 'Missing' | 'Invalid';

export const AGE_GROUP_ORDER: AgeGroup[] = ['≤25', '26–35', '36–45', '46–55', '56+'];

export function categorizeAge(rawAge: unknown): AgeGroup {
  if (rawAge === null || rawAge === undefined || rawAge === '' || rawAge === '?') {
    return 'Missing';
  }
  const n = typeof rawAge === 'number' ? rawAge : Number(String(rawAge).trim());
  if (Number.isNaN(n) || !Number.isFinite(n) || n < 0 || n > 120) {
    return 'Invalid';
  }
  if (n <= 25) return '≤25';
  if (n <= 35) return '26–35';
  if (n <= 45) return '36–45';
  if (n <= 55) return '46–55';
  return '56+';
}

export interface DatasetRow {
  [key: string]: string | number;
}

export interface GroupFairnessRow {
  group: string;
  count: number;
  selectionRate: number;
  ciLow: number;
  ciHigh: number;
  avgPredictedProb: number;
  avgModelConfidence: number;
  tpr: number;
  fpr: number;
  fnr: number;
}

export interface AttributeAuditResult {
  attribute: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  dpDiff: number;
  dpRatio: number;
  eoDiff: number;
  eqOddsDiff: number;
  smallGroupWarnings: string[];
  groups: GroupFairnessRow[];
}

export interface ModelMetrics {
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
}

export interface ModelComparisonRow extends ModelMetrics {
  maxDpDiff: number;
  maxEqOddsDiff: number;
}

export interface MitigationRow {
  strategy: string;
  accuracy: number;
  f1Score: number;
  dpDiff: number;
  dpRatio: number;
  eqOddsDiff: number;
}

export interface ProxyScoreRow {
  feature: string;
  associationStrength: number;
  leakageScore: number;
  tier: 'HIGH' | 'MODERATE' | 'LOW';
}

export interface RootCauseRow {
  driver: string;
  contributionPct: number;
  evidence: string;
}

export interface FeatureImportanceRow {
  feature: string;
  importance: number;
  direction: 'Positive' | 'Negative' | 'Mixed';
}

export interface CounterfactualCase {
  recordIndex: number;
  attribute: string;
  originalGroup: string;
  originalProb: number;
  originalPred: number;
  alternateGroup: string;
  alternateProb: number;
  alternatePred: number;
  probDelta: number;
  decisionFlipped: boolean;
}

export interface TestPredictionPoint {
  gender?: string;
  ageGroup?: string;
  race?: string;
  predictedProb: number;
  modelConfidence: number;
  predictedLabel: number;
  actualLabel: number;
}

export interface FullAuditOutput {
  modelMetrics: ModelMetrics;
  modelComparisons: ModelComparisonRow[];
  attributeAudits: AttributeAuditResult[];
  mitigationComparison: MitigationRow[];
  proxyScores: ProxyScoreRow[];
  rootCauses: RootCauseRow[];
  featureImportances: FeatureImportanceRow[];
  counterfactualCases: CounterfactualCase[];
  knnConsistencyScore: number;
  counterfactualFlipRate: number;
  intersectionalRows: GroupFairnessRow[];
  samplePoints: TestPredictionPoint[];
  biasRiskScore: number;
  riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  testCount: number;
}

interface CartNode {
  isLeaf: boolean;
  prob: number;
  featureIdx?: number;
  threshold?: number;
  left?: CartNode;
  right?: CartNode;
}

function wilsonCI(successes: number, total: number): [number, number] {
  if (total <= 0) return [0, 0];
  const z = 1.96;
  const p = successes / total;
  const denom = 1 + (z * z) / total;
  const center = (p + (z * z) / (2 * total)) / denom;
  const margin = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)) / denom;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function sigmoid(z: number): number {
  const clamped = Math.max(-20, Math.min(20, z));
  return 1 / (1 + Math.exp(-clamped));
}

function computeAuc(yTrue: number[], yProb: number[]): number {
  const pairs = yTrue.map((y, idx) => ({ y, p: yProb[idx] })).sort((a, b) => a.p - b.p);
  let posCount = 0;
  let negCount = 0;
  let rankSum = 0;
  pairs.forEach((item, i) => {
    if (item.y === 1) {
      posCount++;
      rankSum += i + 1;
    } else {
      negCount++;
    }
  });
  if (posCount === 0 || negCount === 0) return 0.5;
  return (rankSum - (posCount * (posCount + 1)) / 2) / (posCount * negCount);
}

function evalClassification(
  algorithm: string,
  yTrue: number[],
  yPred: number[],
  yProb: number[],
): ModelMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else fn++;
  }
  const accuracy = (tp + tn) / Math.max(1, yTrue.length);
  const precision = tp / Math.max(1, tp + fp);
  const recall = tp / Math.max(1, tp + fn);
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const rocAuc = computeAuc(yTrue, yProb);
  return { algorithm, accuracy, precision, recall, f1Score, rocAuc };
}

function giniImpurity(posCount: number, totalCount: number): number {
  if (totalCount === 0) return 0;
  const p = posCount / totalCount;
  return 1 - p * p - (1 - p) * (1 - p);
}

function fitCartTree(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minLeaf: number,
  featureSubset?: number[],
  featureGains?: number[],
): CartNode {
  const n = indices.length;
  let pos = 0;
  for (const idx of indices) pos += y[idx];
  const prob = n > 0 ? pos / n : 0.5;

  if (depth >= maxDepth || n < minLeaf * 2 || pos === 0 || pos === n) {
    return { isLeaf: true, prob };
  }

  const parentGini = giniImpurity(pos, n);
  let bestGain = 0.002;
  let bestFeat = -1;
  let bestThresh = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  const dims = featureSubset || Array.from({ length: X[0].length }, (_, i) => i);

  for (const f of dims) {
    // Sample candidate thresholds across quantiles for O(N) speed
    const vals = indices.map((i) => X[i][f]).sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(vals.length / 8));
    const candidates: number[] = [];
    for (let k = step; k < vals.length - step; k += step) {
      const mid = (vals[k] + vals[k - 1]) / 2;
      if (candidates.length === 0 || candidates[candidates.length - 1] !== mid) {
        candidates.push(mid);
      }
    }
    if (candidates.length === 0 && vals.length > 1) {
      candidates.push((vals[0] + vals[vals.length - 1]) / 2);
    }

    for (const thresh of candidates) {
      const left: number[] = [];
      const right: number[] = [];
      let leftPos = 0;
      for (const idx of indices) {
        if (X[idx][f] <= thresh) {
          left.push(idx);
          leftPos += y[idx];
        } else {
          right.push(idx);
        }
      }
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const rightPos = pos - leftPos;
      const weightedGini =
        (left.length / n) * giniImpurity(leftPos, left.length) +
        (right.length / n) * giniImpurity(rightPos, right.length);
      const gain = parentGini - weightedGini;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = thresh;
        bestLeft = left;
        bestRight = right;
      }
    }
  }

  if (bestFeat === -1) {
    return { isLeaf: true, prob };
  }
  if (featureGains) {
    featureGains[bestFeat] += bestGain * n;
  }

  return {
    isLeaf: false,
    prob,
    featureIdx: bestFeat,
    threshold: bestThresh,
    left: fitCartTree(X, y, bestLeft, depth + 1, maxDepth, minLeaf, featureSubset, featureGains),
    right: fitCartTree(X, y, bestRight, depth + 1, maxDepth, minLeaf, featureSubset, featureGains),
  };
}

function predictCart(node: CartNode, row: number[]): number {
  let cur = node;
  while (!cur.isLeaf && cur.featureIdx !== undefined && cur.threshold !== undefined) {
    cur = row[cur.featureIdx] <= cur.threshold ? cur.left! : cur.right!;
  }
  return cur.prob;
}

function evaluateAttribute(
  attribute: string,
  yTrue: number[],
  yPred: number[],
  yProb: number[],
  groups: string[],
  orderAge = false,
): AttributeAuditResult {
  const map = new Map<
    string,
    { n: number; predPos: number; probSum: number; confSum: number; tp: number; pos: number; fp: number; neg: number }
  >();
  for (let i = 0; i < yTrue.length; i++) {
    const g = groups[i] || 'Missing';
    const cur = map.get(g) || { n: 0, predPos: 0, probSum: 0, confSum: 0, tp: 0, pos: 0, fp: 0, neg: 0 };
    const p = yProb[i] ?? (yPred[i] === 1 ? 1 : 0);
    cur.n++;
    cur.probSum += p;
    cur.confSum += Math.max(p, 1 - p);
    if (yPred[i] === 1) cur.predPos++;
    if (yTrue[i] === 1) {
      cur.pos++;
      if (yPred[i] === 1) cur.tp++;
    } else {
      cur.neg++;
      if (yPred[i] === 1) cur.fp++;
    }
    map.set(g, cur);
  }

  let keys = Array.from(map.keys());
  if (orderAge) {
    const order = [...AGE_GROUP_ORDER, 'Missing', 'Invalid'];
    keys = keys.sort((a, b) => order.indexOf(a as AgeGroup) - order.indexOf(b as AgeGroup));
  } else {
    keys = keys.sort();
  }

  const smallGroupWarnings: string[] = [];
  const rows: GroupFairnessRow[] = keys.map((grp) => {
    const d = map.get(grp)!;
    if (d.n < 30) {
      smallGroupWarnings.push(`Group '${grp}' has ${d.n} samples (< 30); interval uncertainty is elevated.`);
    }
    const sel = d.predPos / Math.max(1, d.n);
    const [ciLow, ciHigh] = wilsonCI(d.predPos, d.n);
    const avgPredictedProb = d.probSum / Math.max(1, d.n);
    const avgModelConfidence = d.confSum / Math.max(1, d.n);
    const tpr = d.pos > 0 ? d.tp / d.pos : 0;
    const fpr = d.neg > 0 ? d.fp / d.neg : 0;
    const fnr = 1 - tpr;
    return {
      group: grp,
      count: d.n,
      selectionRate: sel,
      ciLow,
      ciHigh,
      avgPredictedProb,
      avgModelConfidence,
      tpr,
      fpr,
      fnr,
    };
  });

  const sels = rows.map((r) => r.selectionRate);
  const tprs = rows.map((r) => r.tpr);
  const fprs = rows.map((r) => r.fpr);
  const dpDiff = rows.length ? Math.max(...sels) - Math.min(...sels) : 0;
  const maxSel = rows.length ? Math.max(...sels) : 1;
  const dpRatio = maxSel > 0 && rows.length ? Math.min(...sels) / maxSel : 1;
  const eoDiff = rows.length ? Math.max(...tprs) - Math.min(...tprs) : 0;
  const fprDiff = rows.length ? Math.max(...fprs) - Math.min(...fprs) : 0;
  const eqOddsDiff = Math.max(eoDiff, fprDiff);

  let status: 'PASS' | 'WARN' | 'FAIL' = 'PASS';
  if (dpDiff > 0.10 || eqOddsDiff > 0.10 || dpRatio < 0.80) status = 'FAIL';
  else if (dpDiff > 0.05 || eqOddsDiff > 0.05 || dpRatio < 0.85) status = 'WARN';

  return { attribute, status, dpDiff, dpRatio, eoDiff, eqOddsDiff, smallGroupWarnings, groups: rows };
}

export function runClientSideAudit(
  rows: DatasetRow[],
  targetColumn: string,
  algorithm: 'Logistic Regression' | 'Decision Tree' | 'Random Forest',
  includeSensitive: boolean,
): FullAuditOutput {
  if (rows.length < 40) {
    throw new Error('At least 40 rows are required to run an audit.');
  }

  // Deterministic 75/25 split
  const trainRows: DatasetRow[] = [];
  const testRows: DatasetRow[] = [];
  rows.forEach((r, idx) => {
    if (idx % 4 === 0) testRows.push(r);
    else trainRows.push(r);
  });

  const allCols = Object.keys(rows[0]).filter(
    (c) => c !== targetColumn && !c.endsWith('_id') && c !== 'age_group',
  );
  const sensitiveSet = new Set(['gender', 'sex', 'race', 'age']);
  const modelCols = includeSensitive
    ? allCols
    : allCols.filter((c) => !sensitiveSet.has(c.toLowerCase()));

  const numericCols = modelCols.filter((c) => typeof rows[0][c] === 'number');
  const catCols = modelCols.filter((c) => typeof rows[0][c] !== 'number');

  const means: Record<string, number> = {};
  const stds: Record<string, number> = {};
  for (const col of numericCols) {
    const vals = trainRows.map((r) => Number(r[col]) || 0);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, vals.length);
    means[col] = mean;
    stds[col] = Math.sqrt(variance) || 1;
  }

  const catVocab: Record<string, string[]> = {};
  const encodedFeatureNames: string[] = [...numericCols];
  for (const col of catCols) {
    const uniq = Array.from(new Set(trainRows.map((r) => String(r[col])))).slice(0, 6);
    catVocab[col] = uniq;
    for (const tok of uniq) {
      encodedFeatureNames.push(`${col}=${tok}`);
    }
  }

  const encode = (r: DatasetRow): number[] => {
    const vec: number[] = [];
    for (const col of numericCols) {
      vec.push(((Number(r[col]) || 0) - means[col]) / stds[col]);
    }
    for (const col of catCols) {
      const val = String(r[col]);
      for (const tok of catVocab[col]) {
        vec.push(val === tok ? 1 : 0);
      }
    }
    return vec;
  };

  const XTrain = trainRows.map(encode);
  const yTrain = trainRows.map((r) => (Number(r[targetColumn]) === 1 ? 1 : 0));
  const XTest = testRows.map(encode);
  const yTest = testRows.map((r) => (Number(r[targetColumn]) === 1 ? 1 : 0));

  const dim = Math.max(1, XTrain[0].length);
  const fitLogistic = (sampleWeights?: number[]): { weights: number[]; bias: number } => {
    const weights = new Array(dim).fill(0);
    let bias = 0;
    const lr = 0.16;
    const epochs = 55;
    for (let ep = 0; ep < epochs; ep++) {
      const gradW = new Array(dim).fill(0);
      let gradB = 0;
      for (let i = 0; i < XTrain.length; i++) {
        let z = bias;
        for (let d = 0; d < dim; d++) z += weights[d] * XTrain[i][d];
        const err = (sigmoid(z) - yTrain[i]) * (sampleWeights ? sampleWeights[i] : 1);
        gradB += err;
        for (let d = 0; d < dim; d++) gradW[d] += err * XTrain[i][d];
      }
      const n = XTrain.length;
      bias -= (lr * gradB) / n;
      for (let d = 0; d < dim; d++) {
        weights[d] -= lr * (gradW[d] / n + 0.001 * weights[d]);
      }
    }
    return { weights, bias };
  };

  const predictProbs = (weights: number[], bias: number, matrix: number[][]): number[] =>
    matrix.map((row) => {
      let z = bias;
      for (let d = 0; d < dim; d++) z += weights[d] * row[d];
      return sigmoid(z);
    });

  // Train all 3 models on identical train/test split for direct side-by-side comparison
  const lrFit = fitLogistic();
  const lrProbs = predictProbs(lrFit.weights, lrFit.bias, XTest);

  const allTrainIndices = Array.from({ length: XTrain.length }, (_, i) => i);
  const treeGains = new Array(dim).fill(0);
  const cartTree = fitCartTree(XTrain, yTrain, allTrainIndices, 0, 4, 15, undefined, treeGains);
  const dtProbs = XTest.map((row) => predictCart(cartTree, row));

  const nTrees = 7;
  const rfGains = new Array(dim).fill(0);
  const rfProbs = new Array(XTest.length).fill(0);
  const mTry = Math.max(2, Math.floor(Math.sqrt(dim)));
  for (let t = 0; t < nTrees; t++) {
    const bootIdx: number[] = [];
    for (let i = 0; i < XTrain.length; i++) {
      bootIdx.push((i * 7 + t * 13) % XTrain.length);
    }
    const featSub: number[] = [];
    for (let f = 0; f < mTry; f++) {
      featSub.push((f * 3 + t * 2) % dim);
    }
    const uniqueSub = Array.from(new Set(featSub));
    const tree = fitCartTree(XTrain, yTrain, bootIdx, 0, 4, 12, uniqueSub, rfGains);
    XTest.forEach((row, idx) => {
      rfProbs[idx] += predictCart(tree, row) / nTrees;
    });
  }

  const genderCol = Object.keys(rows[0]).find((c) => c.toLowerCase() === 'gender' || c.toLowerCase() === 'sex');
  const raceCol = Object.keys(rows[0]).find((c) => c.toLowerCase() === 'race');
  const ageCol = Object.keys(rows[0]).find((c) => c.toLowerCase() === 'age');

  const buildModelSummary = (name: string, probs: number[]): ModelComparisonRow => {
    const preds = probs.map((p) => (p >= 0.5 ? 1 : 0));
    const perf = evalClassification(name, yTest, preds, probs);
    const audits: AttributeAuditResult[] = [];
    if (genderCol) audits.push(evaluateAttribute('Gender', yTest, preds, probs, testRows.map((r) => String(r[genderCol]))));
    if (ageCol) audits.push(evaluateAttribute('Age Group', yTest, preds, probs, testRows.map((r) => categorizeAge(r[ageCol])), true));
    if (raceCol) audits.push(evaluateAttribute('Race', yTest, preds, probs, testRows.map((r) => String(r[raceCol]))));
    return {
      ...perf,
      maxDpDiff: audits.length ? Math.max(...audits.map((a) => a.dpDiff)) : 0,
      maxEqOddsDiff: audits.length ? Math.max(...audits.map((a) => a.eqOddsDiff)) : 0,
    };
  };

  const modelComparisons: ModelComparisonRow[] = [
    buildModelSummary('Logistic Regression', lrProbs),
    buildModelSummary('Decision Tree', dtProbs),
    buildModelSummary('Random Forest', rfProbs),
  ];

  const yProb =
    algorithm === 'Logistic Regression'
      ? lrProbs
      : algorithm === 'Decision Tree'
        ? dtProbs
        : rfProbs;

  const yPred = yProb.map((p) => (p >= 0.5 ? 1 : 0));
  const modelMetrics = evalClassification(algorithm, yTest, yPred, yProb);

  // Feature importance from active model
  const rawImportances =
    algorithm === 'Logistic Regression'
      ? lrFit.weights.map((w) => Math.abs(w))
      : algorithm === 'Decision Tree'
        ? treeGains
        : rfGains;
  const totalImp = rawImportances.reduce((a, b) => a + b, 0) || 1;
  const featureImportances: FeatureImportanceRow[] = encodedFeatureNames
    .map((feat, idx) => ({
      feature: feat,
      importance: Number(((rawImportances[idx] / totalImp) * 100).toFixed(2)),
      direction:
        algorithm === 'Logistic Regression'
          ? ((lrFit.weights[idx] >= 0 ? 'Positive' : 'Negative') as 'Positive' | 'Negative')
          : ('Mixed' as const),
    }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);

  const attributeAudits: AttributeAuditResult[] = [];
  if (genderCol) {
    attributeAudits.push(
      evaluateAttribute('Gender', yTest, yPred, yProb, testRows.map((r) => String(r[genderCol]))),
    );
  }
  if (ageCol) {
    attributeAudits.push(
      evaluateAttribute('Age Group', yTest, yPred, yProb, testRows.map((r) => categorizeAge(r[ageCol])), true),
    );
  }
  if (raceCol) {
    attributeAudits.push(
      evaluateAttribute('Race', yTest, yPred, yProb, testRows.map((r) => String(r[raceCol]))),
    );
  }

  // Individual Fairness: k-NN consistency & counterfactual swap
  const evalSubCount = Math.min(80, testRows.length);
  let consistencySum = 0;
  for (let i = 0; i < evalSubCount; i++) {
    const dists: { idx: number; d: number }[] = [];
    for (let j = 0; j < evalSubCount; j++) {
      if (i === j) continue;
      let dist = 0;
      for (let d = 0; d < Math.min(numericCols.length, dim); d++) {
        dist += (XTest[i][d] - XTest[j][d]) ** 2;
      }
      dists.push({ idx: j, d: dist });
    }
    dists.sort((a, b) => a.d - b.d);
    const kNbrs = dists.slice(0, 5);
    const nbrMean = kNbrs.reduce((acc, item) => acc + yProb[item.idx], 0) / Math.max(1, kNbrs.length);
    consistencySum += 1 - Math.abs(yProb[i] - nbrMean);
  }
  const knnConsistencyScore = Number((consistencySum / Math.max(1, evalSubCount)).toFixed(4));

  const primaryAttrName = genderCol || raceCol || ageCol || '';
  const predictSingle = (row: DatasetRow): number => {
    const vec = encode(row);
    if (algorithm === 'Logistic Regression') return predictProbs(lrFit.weights, lrFit.bias, [vec])[0];
    if (algorithm === 'Decision Tree') return predictCart(cartTree, vec);
    return lrProbs[0]; // overridden below cleanly for RF
  };
  void predictSingle;

  const predictRowActive = (row: DatasetRow): number => {
    const vec = encode(row);
    if (algorithm === 'Logistic Regression') return predictProbs(lrFit.weights, lrFit.bias, [vec])[0];
    if (algorithm === 'Decision Tree') return predictCart(cartTree, vec);
    // Re-score RF
    let sum = 0;
    for (let t = 0; t < nTrees; t++) {
      const bootIdx: number[] = [];
      for (let i = 0; i < XTrain.length; i++) bootIdx.push((i * 7 + t * 13) % XTrain.length);
      const featSub: number[] = [];
      for (let f = 0; f < mTry; f++) featSub.push((f * 3 + t * 2) % dim);
      const tree = fitCartTree(XTrain, yTrain, bootIdx, 0, 4, 12, Array.from(new Set(featSub)));
      sum += predictCart(tree, vec) / nTrees;
    }
    return sum;
  };

  const counterfactualCases: CounterfactualCase[] = [];
  let flipCount = 0;
  if (primaryAttrName) {
    const uniqueAlts = Array.from(new Set(testRows.map((r) => String(r[primaryAttrName])))).slice(0, 4);
    const cfCheckLimit = Math.min(40, testRows.length);
    for (let i = 0; i < cfCheckLimit; i++) {
      const origRow = testRows[i];
      const origVal = String(origRow[primaryAttrName]);
      const origProb = yProb[i];
      const origPred = yPred[i];
      let bestAlt = origVal;
      let bestAltProb = origProb;
      let maxDelta = -1;
      let flipped = false;

      if (includeSensitive && uniqueAlts.length > 1) {
        for (const alt of uniqueAlts) {
          if (alt === origVal) continue;
          const swapped = { ...origRow, [primaryAttrName]: alt };
          const altProb = predictRowActive(swapped);
          const delta = Math.abs(altProb - origProb);
          if (delta > maxDelta) {
            maxDelta = delta;
            bestAlt = alt;
            bestAltProb = altProb;
          }
          if ((altProb >= 0.5 ? 1 : 0) !== origPred) {
            flipped = true;
          }
        }
      }
      if (flipped) flipCount++;
      if (i < 15) {
        counterfactualCases.push({
          recordIndex: i,
          attribute: primaryAttrName,
          originalGroup: origVal,
          originalProb: origProb,
          originalPred: origPred,
          alternateGroup: includeSensitive ? bestAlt : 'Excluded from training',
          alternateProb: bestAltProb,
          alternatePred: bestAltProb >= 0.5 ? 1 : 0,
          probDelta: bestAltProb - origProb,
          decisionFlipped: flipped,
        });
      }
    }
  }
  const counterfactualFlipRate = Number((flipCount / Math.max(1, Math.min(40, testRows.length))).toFixed(4));

  // Mitigation comparison on primary sensitive attribute
  const testPrimaryGroups = testRows.map((r) =>
    primaryAttrName === ageCol ? categorizeAge(r[ageCol!]) : String(r[primaryAttrName] || 'Unknown'),
  );
  const trainPrimaryGroups = trainRows.map((r) =>
    primaryAttrName === ageCol ? categorizeAge(r[ageCol!]) : String(r[primaryAttrName] || 'Unknown'),
  );

  const baseAudit = evaluateAttribute(primaryAttrName || 'Group', yTest, yPred, yProb, testPrimaryGroups);

  // Strategy 1: Reweighing
  const groupCounts: Record<string, number> = {};
  const labelCounts: Record<number, number> = { 0: 0, 1: 0 };
  const jointCounts: Record<string, number> = {};
  trainPrimaryGroups.forEach((g, idx) => {
    const y = yTrain[idx];
    groupCounts[g] = (groupCounts[g] || 0) + 1;
    labelCounts[y] = (labelCounts[y] || 0) + 1;
    const k = `${g}|${y}`;
    jointCounts[k] = (jointCounts[k] || 0) + 1;
  });
  const nTr = trainRows.length;
  const reweights = trainPrimaryGroups.map((g, idx) => {
    const y = yTrain[idx];
    const pG = groupCounts[g] / nTr;
    const pY = labelCounts[y] / nTr;
    const pGY = jointCounts[`${g}|${y}`] / nTr;
    return pGY > 0 ? (pG * pY) / pGY : 1;
  });
  const rwFit = fitLogistic(reweights);
  const rwProb = predictProbs(rwFit.weights, rwFit.bias, XTest);
  const rwPred = rwProb.map((p) => (p >= 0.5 ? 1 : 0));
  const rwPerf = evalClassification('Reweighing', yTest, rwPred, rwProb);
  const rwAudit = evaluateAttribute(primaryAttrName, yTest, rwPred, rwProb, testPrimaryGroups);

  // Strategy 2: Group Threshold Adjustment
  const overallSelRate = yPred.filter((v) => v === 1).length / yPred.length;
  const grpThresholds: Record<string, number> = {};
  for (const grp of Object.keys(groupCounts)) {
    const grpProbs = yProb.filter((_, i) => testPrimaryGroups[i] === grp).sort((a, b) => a - b);
    if (grpProbs.length > 0) {
      const qIdx = Math.min(grpProbs.length - 1, Math.max(0, Math.floor((1 - overallSelRate) * grpProbs.length)));
      grpThresholds[grp] = grpProbs[qIdx];
    } else {
      grpThresholds[grp] = 0.5;
    }
  }
  const taPred = yProb.map((p, idx) => (p >= (grpThresholds[testPrimaryGroups[idx]] ?? 0.5) ? 1 : 0));
  const taPerf = evalClassification('Threshold Adjustment', yTest, taPred, yProb);
  const taAudit = evaluateAttribute(primaryAttrName, yTest, taPred, yProb, testPrimaryGroups);

  const mitigationComparison: MitigationRow[] = [
    {
      strategy: 'Baseline (Unmitigated)',
      accuracy: modelMetrics.accuracy,
      f1Score: modelMetrics.f1Score,
      dpDiff: baseAudit.dpDiff,
      dpRatio: baseAudit.dpRatio,
      eqOddsDiff: baseAudit.eqOddsDiff,
    },
    {
      strategy: 'Reweighing (Pre-processing)',
      accuracy: rwPerf.accuracy,
      f1Score: rwPerf.f1Score,
      dpDiff: rwAudit.dpDiff,
      dpRatio: rwAudit.dpRatio,
      eqOddsDiff: rwAudit.eqOddsDiff,
    },
    {
      strategy: 'Threshold Adjustment (Post-processing)',
      accuracy: taPerf.accuracy,
      f1Score: taPerf.f1Score,
      dpDiff: taAudit.dpDiff,
      dpRatio: taAudit.dpRatio,
      eqOddsDiff: taAudit.eqOddsDiff,
    },
  ];

  // Proxy leakage analysis
  const proxyScores: ProxyScoreRow[] = allCols
    .filter((c) => !sensitiveSet.has(c.toLowerCase()))
    .map((col) => {
      let assoc = 0;
      if (typeof rows[0][col] === 'number') {
        const vals = rows.map((r) => Number(r[col]) || 0);
        const totalMean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const ssTotal = vals.reduce((a, b) => a + (b - totalMean) ** 2, 0);
        const byGrp = new Map<string, number[]>();
        rows.forEach((r, idx) => {
          const g = String(r[primaryAttrName] || 'Missing');
          const arr = byGrp.get(g) || [];
          arr.push(vals[idx]);
          byGrp.set(g, arr);
        });
        let ssBetween = 0;
        byGrp.forEach((arr) => {
          const m = arr.reduce((a, b) => a + b, 0) / arr.length;
          ssBetween += arr.length * (m - totalMean) ** 2;
        });
        assoc = ssTotal > 0 ? Math.sqrt(ssBetween / ssTotal) : 0;
      } else {
        const groups = Array.from(new Set(rows.map((r) => String(r[primaryAttrName] || 'Missing'))));
        const cats = Array.from(new Set(rows.map((r) => String(r[col] || 'Missing'))));
        if (groups.length > 1 && cats.length > 1) {
          let tvd = 0;
          const baseCounts: Record<string, number> = {};
          rows.forEach((r) => {
            const c = String(r[col]);
            baseCounts[c] = (baseCounts[c] || 0) + 1 / rows.length;
          });
          groups.forEach((g) => {
            const sub = rows.filter((r) => String(r[primaryAttrName]) === g);
            if (sub.length === 0) return;
            cats.forEach((c) => {
              const p = sub.filter((r) => String(r[col]) === c).length / sub.length;
              tvd += Math.abs(p - (baseCounts[c] || 0)) / groups.length;
            });
          });
          assoc = Math.min(1, tvd * 1.4);
        }
      }
      const leakageScore = Math.min(100, Number((assoc * 100).toFixed(1)));
      const tier: 'HIGH' | 'MODERATE' | 'LOW' =
        leakageScore >= 35 ? 'HIGH' : leakageScore >= 15 ? 'MODERATE' : 'LOW';
      return { feature: col, associationStrength: Number(assoc.toFixed(3)), leakageScore, tier };
    })
    .sort((a, b) => b.leakageScore - a.leakageScore);

  // Root cause attribution
  const labelRates = new Map<string, { pos: number; n: number }>();
  testPrimaryGroups.forEach((g, i) => {
    const cur = labelRates.get(g) || { pos: 0, n: 0 };
    cur.n++;
    if (yTest[i] === 1) cur.pos++;
    labelRates.set(g, cur);
  });
  const baseRates = Array.from(labelRates.values()).map((v) => v.pos / Math.max(1, v.n));
  const labelGap = baseRates.length > 1 ? Math.max(...baseRates) - Math.min(...baseRates) : 0;
  const shares = Array.from(labelRates.values()).map((v) => v.n / testRows.length);
  const repSkew = shares.length > 1 ? (1 - Math.min(...shares) / Math.max(...shares)) * 100 : 0;
  const maxProxy = proxyScores[0]?.leakageScore || 0;
  const modelAmp = Math.max(0, baseAudit.dpDiff - labelGap) * 200 + baseAudit.eqOddsDiff * 180;
  const labelScore = labelGap * 240;
  const totalRc = Math.max(1, repSkew + maxProxy + modelAmp + labelScore);

  const rootCauses: RootCauseRow[] = [
    {
      driver: 'Historical Label Disparity',
      contributionPct: Number(((labelScore / totalRc) * 100).toFixed(1)),
      evidence: `Ground-truth positive base rate varies by ${(labelGap * 100).toFixed(1)}% across ${primaryAttrName}.`,
    },
    {
      driver: 'Model Error & Threshold Disparity',
      contributionPct: Number(((modelAmp / totalRc) * 100).toFixed(1)),
      evidence: `Equalized Odds diff is ${baseAudit.eqOddsDiff.toFixed(3)}; DP diff is ${baseAudit.dpDiff.toFixed(3)}.`,
    },
    {
      driver: 'Proxy Feature Leakage',
      contributionPct: Number(((maxProxy / totalRc) * 100).toFixed(1)),
      evidence: proxyScores[0]
        ? `Top correlated feature '${proxyScores[0].feature}' scored ${proxyScores[0].leakageScore}/100.`
        : 'No proxy features evaluated.',
    },
    {
      driver: 'Representation Imbalance',
      contributionPct: Number(((repSkew / totalRc) * 100).toFixed(1)),
      evidence: `Smallest group accounts for ${(Math.min(...shares) * 100).toFixed(1)}% of test records.`,
    },
  ].sort((a, b) => b.contributionPct - a.contributionPct);

  const interGroups = testRows.map((r) => {
    const g = genderCol ? String(r[genderCol]) : 'All';
    const a = ageCol ? categorizeAge(r[ageCol]) : 'All';
    return `${g} | ${a}`;
  });
  const interAudit = evaluateAttribute('Gender × Age Group', yTest, yPred, yProb, interGroups);

  const samplePoints: TestPredictionPoint[] = testRows.map((r, i) => ({
    gender: genderCol ? String(r[genderCol]) : undefined,
    ageGroup: ageCol ? categorizeAge(r[ageCol]) : undefined,
    race: raceCol ? String(r[raceCol]) : undefined,
    predictedProb: yProb[i],
    modelConfidence: Math.max(yProb[i], 1 - yProb[i]),
    predictedLabel: yPred[i],
    actualLabel: yTest[i],
  }));

  const peakDp = Math.max(...attributeAudits.map((a) => a.dpDiff), 0);
  const peakEqOdds = Math.max(...attributeAudits.map((a) => a.eqOddsDiff), 0);
  const riskRaw =
    Math.min(100, (peakDp / 0.25) * 100) * 0.4 +
    Math.min(100, (peakEqOdds / 0.25) * 100) * 0.4 +
    Math.min(100, maxProxy) * 0.2;
  const biasRiskScore = Number(Math.min(100, riskRaw).toFixed(1));
  const riskLevel =
    biasRiskScore >= 70 ? 'CRITICAL' : biasRiskScore >= 45 ? 'HIGH' : biasRiskScore >= 20 ? 'MODERATE' : 'LOW';

  return {
    modelMetrics,
    modelComparisons,
    attributeAudits,
    mitigationComparison,
    proxyScores,
    rootCauses,
    featureImportances,
    counterfactualCases,
    knnConsistencyScore,
    counterfactualFlipRate,
    intersectionalRows: interAudit.groups,
    samplePoints,
    biasRiskScore,
    riskLevel,
    testCount: testRows.length,
  };
}
