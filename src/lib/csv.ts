export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededShuffle<T>(arr: T[], seed: number): T[] {
  const copy = [...arr];
  const rng = mulberry32(seed);
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function parseCsv(rawText: string): {
  headers: string[];
  rows: Record<string, string>[];
} {
  const clean = rawText.replace(/^\uFEFF/, '').trim();
  const lines = clean.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };

  const parseLine = (line: string): string[] => {
    const out: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        out.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    out.push(current.trim());
    return out;
  };

  const headers = parseLine(lines[0]);
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = parseLine(lines[i]);
    if (vals.length !== headers.length) continue;
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      row[headers[j]] = vals[j];
    }
    rows.push(row);
  }
  return { headers, rows };
}

export function trainTestSplitIndices(
  total: number,
  testSize: number,
  seed: number,
  stratifyLabels?: number[]
): { trainIdx: number[]; testIdx: number[] } {
  if (!stratifyLabels) {
    const indices = seededShuffle(
      Array.from({ length: total }, (_, i) => i),
      seed
    );
    const nTest = Math.max(1, Math.round(total * testSize));
    return {
      testIdx: indices.slice(0, nTest),
      trainIdx: indices.slice(nTest),
    };
  }

  const byClass = new Map<number, number[]>();
  for (let i = 0; i < total; i++) {
    const lbl = stratifyLabels[i];
    if (!byClass.has(lbl)) byClass.set(lbl, []);
    byClass.get(lbl)!.push(i);
  }

  const trainIdx: number[] = [];
  const testIdx: number[] = [];
  let classOffset = 0;
  for (const [, classIndices] of byClass.entries()) {
    const shuffled = seededShuffle(classIndices, seed + classOffset * 97);
    const nTest = Math.max(1, Math.round(shuffled.length * testSize));
    testIdx.push(...shuffled.slice(0, nTest));
    trainIdx.push(...shuffled.slice(nTest));
    classOffset++;
  }

  return {
    trainIdx: seededShuffle(trainIdx, seed + 17),
    testIdx: seededShuffle(testIdx, seed + 31),
  };
}
