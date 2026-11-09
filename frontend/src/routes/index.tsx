import { createFileRoute, Link, useBlocker } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { Check, FileText, LoaderCircle, RotateCcw, Sparkles, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { CandidateCard } from "@/features/screening/components/candidate-card";
import { cn } from "@/lib/utils";
import {
  average,
  EMPLOYMENT,
  formatSize,
  levelOf,
  useScreening,
} from "@/features/screening/screening";
import {
  isJobDescriptionValid,
  jobDescriptionLength,
  JOB_DESCRIPTION_MAX_LENGTH,
  JOB_DESCRIPTION_MIN_LENGTH,
} from "@/features/screening/job-description";

export const Route = createFileRoute("/")({
  component: ScreenPage,
});

function Step({
  n,
  title,
  done,
  children,
  stretchBody = false,
}: {
  n: number;
  title: string;
  done: boolean;
  children: React.ReactNode;
  stretchBody?: boolean;
}) {
  return (
    <section
      className={cn(
        "rounded-3xl border bg-card p-6 shadow-soft",
        stretchBody && "lg:flex lg:min-h-[320px] lg:flex-col",
      )}
    >
      <h2 className="mb-4 flex items-center gap-3 font-display text-xl font-semibold">
        <span
          className={cn(
            "grid size-8 place-items-center rounded-full text-sm font-bold transition-colors",
            done ? "bg-success text-primary-foreground" : "bg-secondary text-secondary-foreground",
          )}
        >
          {done ? <Check className="size-4" /> : n}
        </span>
        {title}
      </h2>
      {stretchBody ? (
        <div className="lg:flex lg:min-h-0 lg:flex-1 lg:flex-col">{children}</div>
      ) : (
        children
      )}
    </section>
  );
}

function ScreenPage() {
  const {
    jd,
    setJD,
    title,
    setTitle,
    employment,
    setEmployment,
    files,
    setFiles,
    candidates,
    runScreening,
    newJob,
    activeJob,
    jobs,
  } = useScreening();
  const input = useRef<HTMLInputElement>(null);
  const employmentOptions = useRef<Array<HTMLButtonElement | null>>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [filter, setFilter] = useState<"all" | "great" | "good" | "low">("all");
  const [resultsJD, setResultsJD] = useState(jd);
  const [failedRun, setFailedRun] = useState(false);
  const [submitFailed, setSubmitFailed] = useState(false);
  const jdLength = jobDescriptionLength(jd);
  const showMinimumLengthError =
    (jdLength > 0 && jdLength < JOB_DESCRIPTION_MIN_LENGTH) ||
    (submitFailed && jdLength < JOB_DESCRIPTION_MIN_LENGTH);
  const savedJob = jobs.find((job) => job.id === activeJob);
  const matchesSavedJob =
    savedJob !== undefined &&
    savedJob.title === title &&
    savedJob.employment === employment &&
    savedJob.jd === jd &&
    savedJob.files.length === files.length &&
    savedJob.files.every((file, index) => file.id === files[index]?.id);
  const hasUnsavedInput =
    !matchesSavedJob && (title.trim().length > 0 || jd.trim().length > 0 || files.length > 0);

  useBlocker({
    shouldBlockFn: () =>
      hasUnsavedInput && !window.confirm("You have unsaved input. Leave the Screen page?"),
    enableBeforeUnload: hasUnsavedInput,
    disabled: !hasUnsavedInput,
  });

  function add(list: File[]) {
    const ok = list.filter(
      (f) => f.name.toLowerCase().endsWith(".pdf") && f.size <= 10 * 1024 * 1024,
    );
    if (ok.length !== list.length) toast.error("Only PDF files up to 10 MB");
    setFiles((cur) => [
      ...cur,
      ...ok
        .filter((f) => !cur.some((c) => c.name === f.name))
        .map((f) => ({ id: crypto.randomUUID(), name: f.name, size: f.size, file: f })),
    ]);
  }

  async function run() {
    setBusy(true);
    setProgress(0);
    setSubmitFailed(false);
    try {
      const results = await runScreening((d, t) => setProgress(Math.round((d / t) * 100)));
      setResultsJD(jd);
      setFailedRun(false);
      const failures = results.filter((candidate) => candidate.status === "error").length;
      if (failures > 0) {
        toast.warning(
          `${results.length - failures} resumes analyzed; ${failures} could not be analyzed.`,
        );
      } else {
        toast.success(`${results.length} resumes analyzed successfully.`);
      }
      setTimeout(
        () => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }),
        50,
      );
    } catch (error) {
      setSubmitFailed(true);
      if (candidates.length > 0) setFailedRun(true);
      toast.error(
        error instanceof Error ? error.message : "Something went wrong while checking resumes.",
        { duration: 3000, closeButton: true, className: "screen-analysis-error" },
      );
    } finally {
      setBusy(false);
    }
  }

  const ready = isJobDescriptionValid(jd) && files.length > 0;
  const resultsStale = candidates.length > 0 && (failedRun || jd.trim() !== resultsJD.trim());
  const shown = candidates.filter((c) =>
    c.score === null ? filter === "all" : filter === "all" || levelOf(c.score) === filter,
  );
  const top = candidates.find((c) => c.score !== null);

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <div className="py-2 text-center">
        <h1 className="font-display text-4xl font-bold sm:text-5xl">
          Find your best <span className="bg-hero bg-clip-text text-transparent">candidates</span>
        </h1>
      </div>

      <div className="grid items-stretch gap-6 lg:grid-cols-2">
        <Step n={1} title="Describe the job" done={isJobDescriptionValid(jd)} stretchBody>
          <Input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            maxLength={100}
            placeholder="Job title, e.g. Product Designer"
            className="mb-3 h-11 rounded-2xl text-base"
            aria-label="Job title"
          />
          <div className="mb-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Employment type">
            {EMPLOYMENT.map((t, index) => (
              <button
                key={t}
                ref={(element) => {
                  employmentOptions.current[index] = element;
                }}
                type="button"
                role="radio"
                aria-checked={employment === t}
                tabIndex={employment === t ? 0 : -1}
                onClick={() => setEmployment(t)}
                onKeyDown={(event) => {
                  const nextIndex =
                    event.key === "ArrowRight" || event.key === "ArrowDown"
                      ? (index + 1) % EMPLOYMENT.length
                      : event.key === "ArrowLeft" || event.key === "ArrowUp"
                        ? (index - 1 + EMPLOYMENT.length) % EMPLOYMENT.length
                        : event.key === "Home"
                          ? 0
                          : event.key === "End"
                            ? EMPLOYMENT.length - 1
                            : -1;
                  if (nextIndex < 0) return;
                  event.preventDefault();
                  setEmployment(EMPLOYMENT[nextIndex]!);
                  employmentOptions.current[nextIndex]?.focus();
                }}
                className={cn(
                  "rounded-full px-3 py-1.5 text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  employment === t
                    ? "bg-secondary text-secondary-foreground"
                    : "text-muted-foreground hover:bg-muted",
                )}
              >
                {t}
              </button>
            ))}
          </div>
          <Textarea
            value={jd}
            onChange={(e) => setJD(e.target.value)}
            aria-describedby={cn(
              "job-description-counter",
              showMinimumLengthError && "job-description-error",
            )}
            rows={6}
            placeholder="Paste the job description — role, responsibilities and must-have skills…"
            className="rounded-2xl text-base lg:h-24 lg:min-h-0 lg:flex-none"
          />
          <div className="mt-2 flex min-h-4 items-center justify-between gap-2 text-xs text-muted-foreground">
            {showMinimumLengthError ? (
              <p id="job-description-error">Add at least {JOB_DESCRIPTION_MIN_LENGTH} characters</p>
            ) : (
              <span />
            )}
            <p id="job-description-counter" aria-live="polite">
              {jdLength.toLocaleString()} / {JOB_DESCRIPTION_MAX_LENGTH.toLocaleString()}
            </p>
          </div>
        </Step>

        <Step n={2} title="Add resumes" done={files.length > 0} stretchBody>
          <div
            className={cn(
              "relative",
              files.length === 0 && "lg:flex lg:flex-1 lg:flex-col",
              files.length > 0 && "-mr-3 flex items-center justify-between gap-2",
            )}
          >
            <input
              ref={input}
              id="resume-upload"
              type="file"
              accept=".pdf"
              multiple
              aria-label="Upload PDF resumes"
              className="peer sr-only"
              onChange={(e) => {
                add(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
            {files.length > 0 && (
              <p className="min-w-0 truncate whitespace-nowrap pl-[calc(var(--spacing)*3+1px)] text-xs font-semibold">
                {files.length} {files.length === 1 ? "resume" : "resumes"} ·{" "}
                {formatSize(files.reduce((total, file) => total + file.size, 0))}
              </p>
            )}
            <label
              htmlFor="resume-upload"
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                add(Array.from(e.dataTransfer.files));
              }}
              className={cn(
                "block cursor-pointer border-2 border-dashed transition-colors peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2",
                files.length === 0
                  ? "rounded-2xl p-8 text-center lg:flex lg:flex-1 lg:flex-col lg:justify-center"
                  : "flex shrink-0 items-center gap-1.5 rounded-xl px-2 py-2 text-xs",
                dragging
                  ? "border-primary bg-secondary"
                  : "hover:border-primary hover:bg-secondary/50",
              )}
            >
              <UploadCloud
                className={cn("text-primary", files.length === 0 ? "mx-auto size-10" : "size-4")}
              />
              {files.length === 0 ? (
                <>
                  <p className="mt-2 font-semibold">Drop PDF resumes here or click to choose</p>
                  <p className="text-sm text-muted-foreground">Up to 10 MB each</p>
                </>
              ) : (
                <span className="font-bold">Add more PDFs</span>
              )}
            </label>
            {files.length > 0 && (
              <button
                type="button"
                onClick={() => setFiles([])}
                className="shrink-0 rounded-lg px-1 py-1 text-xs font-semibold text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                Clear all
              </button>
            )}
          </div>
          {files.length > 0 && (
            <ul
              className={cn(
                "mt-2 -mr-3 grid h-[11.5rem] grid-cols-1 content-start gap-2 overflow-y-auto pr-3 lg:h-auto lg:min-h-0 lg:flex-1 lg:basis-0",
                files.length >= 6 && "lg:grid-cols-2",
              )}
            >
              {files.map((f) => (
                <li
                  key={f.id}
                  className="flex min-w-0 items-center gap-2 rounded-xl border py-2 pl-3 pr-2 text-sm"
                >
                  <FileText className="size-4 shrink-0 text-primary" />
                  <span className="min-w-0 flex-1 truncate">{f.name}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {formatSize(f.size)}
                  </span>
                  <button
                    type="button"
                    aria-label={`Remove ${f.name}`}
                    onClick={() => setFiles((c) => c.filter((x) => x.id !== f.id))}
                    className="grid size-7 shrink-0 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <X className="size-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Step>
      </div>

      <Step n={3} title="See who fits" done={candidates.length > 0}>
        <Button
          size="lg"
          disabled={!ready || busy}
          onClick={run}
          className="h-14 w-full rounded-2xl bg-hero text-base font-semibold text-primary-foreground hover:opacity-90"
        >
          {busy ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
          {busy ? "Checking resumes…" : candidates.length ? "Check again" : "Find matches"}
        </Button>
        {busy && <Progress value={progress} className="mt-4 h-1" />}
        {!ready && !busy && (
          <p className="mt-2 text-center text-sm text-muted-foreground">
            Finish steps 1 and 2 to continue.
          </p>
        )}
      </Step>

      {busy && (
        <div className="space-y-3 pt-4" aria-label="Loading results">
          {files.slice(0, 4).map((f) => (
            <Skeleton key={f.id} className="h-20 w-full rounded-2xl" />
          ))}
        </div>
      )}

      {!busy && candidates.length > 0 && (
        <section id="results" className="space-y-4 pt-4">
          {resultsStale && (
            <p
              role="status"
              className="rounded-2xl border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-foreground"
            >
              These results are from the previous job description and are stale. A new screening
              must succeed before they update.
            </p>
          )}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {[
              { label: "Resumes scanned", value: String(candidates.length) },
              {
                label: "Average match",
                value: candidates.some((c) => c.score !== null) ? `${average(candidates)}%` : "—",
              },
              { label: "Top candidate", value: top?.name ?? "—" },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border bg-card p-4">
                <p className="text-xs text-muted-foreground">{s.label}</p>
                <p className="mt-1 truncate font-display text-xl font-bold capitalize">{s.value}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <Link to="/shortlist" className="text-sm font-semibold text-primary hover:underline">
              Open full shortlist
            </Link>
            <div className="flex gap-1 rounded-full bg-muted p-1">
              {(
                [
                  ["all", "All"],
                  ["great", "Great"],
                  ["good", "Good"],
                  ["low", "Weak"],
                ] as const
              ).map(([k, label]) => (
                <button
                  key={k}
                  onClick={() => setFilter(k)}
                  className={cn(
                    "rounded-full px-3 py-1.5 text-sm font-semibold transition-colors",
                    filter === k
                      ? "bg-card text-foreground shadow-soft"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          {shown.map((c) => (
            <CandidateCard key={c.id} c={c} rank={candidates.indexOf(c) + 1} />
          ))}
          {shown.length === 0 && (
            <p className="py-6 text-center text-muted-foreground">No candidates in this group.</p>
          )}
          <button
            onClick={() => {
              newJob();
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            className="mx-auto flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground"
          >
            <RotateCcw className="size-4" /> Start a new screening
          </button>
        </section>
      )}
    </main>
  );
}
