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
        <LoginForm />
      </section>
    </main>
  );
}
