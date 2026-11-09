import { describe, expect, it } from "vitest";
import {
  isJobDescriptionValid,
  jobDescriptionLength,
  JOB_DESCRIPTION_MAX_LENGTH,
  JOB_DESCRIPTION_MIN_LENGTH,
} from "@/features/screening/job-description";

describe("job description validation", () => {
  it("accepts trimmed lengths from 50 through 10,000 characters", () => {
    expect(isJobDescriptionValid("x".repeat(JOB_DESCRIPTION_MIN_LENGTH))).toBe(true);
    expect(isJobDescriptionValid("x".repeat(JOB_DESCRIPTION_MAX_LENGTH))).toBe(true);
    expect(isJobDescriptionValid(`  ${"x".repeat(50)}  `)).toBe(true);
  });

  it("rejects descriptions outside the backend limits", () => {
    expect(isJobDescriptionValid("x".repeat(JOB_DESCRIPTION_MIN_LENGTH - 1))).toBe(false);
    expect(isJobDescriptionValid("x".repeat(JOB_DESCRIPTION_MAX_LENGTH + 1))).toBe(false);
    expect(isJobDescriptionValid(`  ${"x".repeat(49)}  `)).toBe(false);
  });

  it("counts Unicode code points like the backend character limit", () => {
    expect(jobDescriptionLength("😀".repeat(50))).toBe(50);
    expect(isJobDescriptionValid("😀".repeat(50))).toBe(true);
  });
});
