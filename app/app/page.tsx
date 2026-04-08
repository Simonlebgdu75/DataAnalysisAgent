import { Suspense } from "react";

import { PeQaShell } from "@/components/pe-qa-shell";

function AppPageFallback() {
  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Vercel BFF + LangGraph</p>
          <h1>Loading `pe_qa`…</h1>
          <p>Rebuilding the client state for the protected conversation view.</p>
        </div>
      </section>
    </main>
  );
}

export default function AppPage() {
  return (
    <Suspense fallback={<AppPageFallback />}>
      <PeQaShell />
    </Suspense>
  );
}
