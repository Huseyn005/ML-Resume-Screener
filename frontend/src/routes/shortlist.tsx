import { createFileRoute, Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { CandidateTable } from "@/features/screening/components/candidate-table";
import { useScreening } from "@/features/screening/screening";

export const Route = createFileRoute("/shortlist")({
  component: ShortlistPage,
});

function ShortlistPage() {
  const { candidates, jobs } = useScreening();
  const empty = candidates.length === 0 && !jobs.some((j) => j.candidates.length);
  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <h1 className="font-display text-3xl font-bold sm:text-4xl">Shortlist</h1>
      {empty ? (
        <div className="rounded-3xl border bg-card px-6 py-16 text-center shadow-soft">
          <p className="text-muted-foreground">
            No candidates yet. Screen some resumes to see them ranked here.
          </p>
          <Button asChild className="mt-5 rounded-full">
            <Link to="/">Screen resumes</Link>
          </Button>
        </div>
      ) : (
        <CandidateTable />
      )}
    </main>
  );
}
