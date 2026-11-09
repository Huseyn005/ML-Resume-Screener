import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const { clearResumes } = vi.hoisted(() => ({ clearResumes: vi.fn() }));
const { toastError, toastSuccess } = vi.hoisted(() => ({
  toastError: vi.fn(),
  toastSuccess: vi.fn(),
}));

vi.mock("@/features/screening/resume-store", () => ({
  clearResumes,
  deleteResumes: vi.fn(),
  getResume: vi.fn(),
  putResume: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    error: toastError,
    success: toastSuccess,
    warning: vi.fn(),
  },
}));

import { ScreeningProvider, useScreening } from "@/features/screening/screening";

afterEach(() => {
  clearResumes.mockReset();
  toastError.mockReset();
  toastSuccess.mockReset();
  localStorage.clear();
});

function wrapper({ children }: { children: ReactNode }) {
  return <ScreeningProvider>{children}</ScreeningProvider>;
}

describe("screening data deletion", () => {
  it("keeps screening data and reports an error when IndexedDB clearing fails", async () => {
    localStorage.clear();
    clearResumes.mockRejectedValueOnce(new Error("storage is unavailable"));

    const { result } = renderHook(() => useScreening(), { wrapper });
    const existingJobs = result.current.jobs;

    await act(async () => {
      await result.current.clearData();
    });

    expect(result.current.jobs).toEqual(existingJobs);
    expect(toastError).toHaveBeenCalledWith(
      "Could not clear stored resumes: storage is unavailable",
    );
    expect(toastSuccess).not.toHaveBeenCalledWith("All screening data cleared");
  });
});
