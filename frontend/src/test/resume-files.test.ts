import { describe, expect, it } from "vitest";
import { filterDuplicateResumeFiles } from "@/features/screening/resume-files";

function file(name: string, size: number, lastModified: number) {
  return new File(["x".repeat(size)], name, { lastModified });
}

describe("filterDuplicateResumeFiles", () => {
  it("skips only files with matching name, size, and last-modified time", () => {
    const existing = [file("Resume.pdf", 1, 100)];
    const incoming = [
      file("Resume.pdf", 1, 100),
      file("Resume.pdf", 2, 100),
      file("Resume.pdf", 1, 101),
      file("Other.pdf", 1, 100),
    ];

    const result = filterDuplicateResumeFiles(existing, incoming);

    expect(result.duplicateCount).toBe(1);
    expect(result.files).toEqual(incoming.slice(1));
  });

  it("counts duplicate files within the same selection", () => {
    const first = file("Resume.pdf", 1, 100);
    const second = file("Resume.pdf", 1, 100);

    const result = filterDuplicateResumeFiles([], [first, second]);

    expect(result.files).toEqual([first]);
    expect(result.duplicateCount).toBe(1);
  });

  it("does not treat legacy entries without a modification time as duplicates", () => {
    const incoming = file("Resume.pdf", 1, 100);

    const result = filterDuplicateResumeFiles(
      [{ name: incoming.name, size: incoming.size }],
      [incoming],
    );

    expect(result.files).toEqual([incoming]);
    expect(result.duplicateCount).toBe(0);
  });
});
