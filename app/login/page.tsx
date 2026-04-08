import { redirect } from "next/navigation";

import { LoginForm } from "@/components/login-form";
import { isCurrentSessionAuthenticated } from "@/lib/auth";

export default async function LoginPage() {
  if (await isCurrentSessionAuthenticated()) {
    redirect("/app");
  }

  return (
    <main className="login-shell">
      <section className="login-card">
        <p className="eyebrow">Shared Demo Gate</p>
        <h1>Unlock `pe_qa`</h1>
        <p>
          This deployment is intentionally gated. Enter the shared password to
          access the demo app and its same-origin BFF routes.
        </p>
        <LoginForm />
      </section>
    </main>
  );
}

