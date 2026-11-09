type ResumeIdentity = { name: string; size: number; lastModified?: number };

function identity(file: ResumeIdentity): string {
  return JSON.stringify([file.name, file.size, file.lastModified]);
}

export function filterDuplicateResumeFiles<T extends ResumeIdentity>(
  existing: ResumeIdentity[],
  incoming: T[],
): { files: T[]; duplicateCount: number } {
  const seen = new Set(existing.map(identity));
  const files: T[] = [];
  let duplicateCount = 0;

  for (const file of incoming) {
    const key = identity(file);
    if (seen.has(key)) {
      duplicateCount += 1;
    } else {
      seen.add(key);
      files.push(file);
    }
  }

  return { files, duplicateCount };
}
