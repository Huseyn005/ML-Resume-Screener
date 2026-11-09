import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useScreening, type Weights } from "@/features/screening/screening";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});

function SettingsPage() {
  const demoMode = import.meta.env["VITE_USE_MOCK"] === "true";
  const { profile, setProfile, weights, setWeights, clearData } = useScreening();
  const [name, setName] = useState(profile.name);
  const [email, setEmail] = useState(profile.email);
  const [draft, setDraft] = useState(weights);
  useEffect(() => {
    setName(profile.name);
    setEmail(profile.email);
  }, [profile]);
  useEffect(() => setDraft(weights), [weights]);
  const total = draft.skills + draft.experience + draft.education;

  return (
    <main className="mx-auto max-w-5xl space-y-8 px-4 py-10">
      <h1 className="font-display text-3xl font-bold sm:text-4xl">Settings</h1>
      <form
        className="grid gap-6 lg:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (demoMode && total !== 100) {
            toast.error("Scoring weights must add up to 100%.");
            return;
          }
          setProfile({ name: name.trim(), email: email.trim() });
          if (demoMode) setWeights(draft);
          toast.success("Settings saved", {
            closeButton: true,
            className: "settings-toast",
          });
        }}
      >
        <section className="flex h-full flex-col gap-4 rounded-3xl border bg-card p-6 shadow-soft">
          <h2 className="font-display text-xl font-semibold">Profile</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm">
              Name
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={100}
                placeholder="Your name"
                className="mt-2 rounded-xl"
              />
            </label>
            <label className="text-sm">
              Email
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                maxLength={255}
                placeholder="you@company.com"
                className="mt-2 rounded-xl"
              />
            </label>
          </div>
          <Button
            className="mt-auto h-12 w-full self-center rounded-full text-base sm:w-48"
            disabled={demoMode && total !== 100}
          >
            Save changes
          </Button>
        </section>
        <section className="flex h-full flex-col gap-5 rounded-3xl border bg-card p-6 shadow-soft">
          <h2 className="font-display text-xl font-semibold">Scoring weights</h2>
          {demoMode ? (
            <>
              {(Object.keys(draft) as (keyof Weights)[]).map((key) => (
                <div key={key}>
                  <div className="mb-3 flex justify-between text-sm">
                    <label htmlFor={`w-${key}`} className="capitalize">
                      {key}
                    </label>
                    <span>{draft[key]}%</span>
                  </div>
                  <Slider
                    id={`w-${key}`}
                    aria-label={`${key} weight`}
                    min={0}
                    max={100}
                    step={5}
                    value={[draft[key]]}
                    onValueChange={(v) => setDraft((p) => ({ ...p, [key]: v[0] ?? 0 }))}
                  />
                </div>
              ))}
              <p className="max-w-[42ch] text-sm leading-relaxed text-muted-foreground">
                Total {total}%{total !== 100 ? " · must equal 100%" : ""}. Applies the next time you
                check resumes.
              </p>
            </>
          ) : (
            <>
              <p className="max-w-[42ch] text-sm leading-relaxed text-muted-foreground">
                Live scores come from the server model. Scoring weights are configured on the
                server.
              </p>
              <Button
                type="button"
                className="mt-auto h-12 w-full self-center rounded-full text-base sm:w-48"
                onClick={() =>
                  toast("Custom scoring controls are planned for a future", {
                    closeButton: true,
                    className: "settings-toast",
                  })
                }
              >
                Coming soon
              </Button>
            </>
          )}
        </section>
      </form>
      <section className="space-y-4 border-t pt-8">
        <h2 className="font-display text-xl font-semibold">Data</h2>
        <p className="text-sm text-muted-foreground">Everything is stored in this browser only.</p>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="outline" className="rounded-full text-destructive">
              <Trash2 />
              Clear all data
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent className="rounded-3xl">
            <AlertDialogTitle>Clear all data?</AlertDialogTitle>
            <AlertDialogDescription>
              All jobs, candidates and resumes will be permanently removed.
            </AlertDialogDescription>
            <AlertDialogFooter>
              <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
              <AlertDialogAction className="rounded-full" onClick={() => void clearData()}>
                Clear data
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </section>
    </main>
  );
}
