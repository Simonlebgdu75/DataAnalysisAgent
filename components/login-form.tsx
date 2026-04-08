"use client";

import { startTransition, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

type ApiErrorResponse = {
  error?: {
    message?: string;
  };
};

export function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, setIsPending] = useState(false);

  const redirectTarget = (() => {
    const raw = searchParams.get("redirect");
    return raw && raw.startsWith("/") ? raw : "/app";
  })();

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (isPending) {
      return;
    }

    setIsPending(true);
    setError(null);

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ password }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | ApiErrorResponse
          | null;

        throw new Error(
          payload?.error?.message ?? "Login failed. Please try again.",
        );
      }

      startTransition(() => {
        router.replace(redirectTarget);
      });
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Login failed. Please try again.",
      );
    } finally {
      setIsPending(false);
    }
  }

  return (
    <form className="login-form" onSubmit={handleSubmit}>
      <label className="login-label" htmlFor="shared-password">
        Shared password
        <input
          id="shared-password"
          className="login-input"
          name="password"
          type="password"
          autoComplete="current-password"
          value={password}
          disabled={isPending}
          onChange={(event) => setPassword(event.target.value)}
        />
      </label>

      {error ? (
        <div className="error-banner" role="alert">
          <h3>Access denied</h3>
          <p>{error}</p>
        </div>
      ) : null}

      <button className="button" type="submit" disabled={isPending}>
        {isPending ? "Opening…" : "Enter demo"}
      </button>
    </form>
  );
}

