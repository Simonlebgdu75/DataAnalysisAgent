import { Suspense } from "react";

import { PeDealShell } from "@/components/pe-deal-shell";

function AppPageFallback() {
  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Vercel BFF + LangGraph</p>
          <h1>Loading `pe_deal`...</h1>
          <p>Rebuilding the client state for the protected conversation view.</p>
        </div>
      </section>
    </main>
  );
}

export default function AppPage() {
  return (
    <Suspense fallback={<AppPageFallback />}>
      <PeDealShell />
    </Suspense>
  );
}
