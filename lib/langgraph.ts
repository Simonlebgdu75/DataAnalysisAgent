import { getLangGraphEnv } from "@/lib/env";
import { RouteError } from "@/lib/request";

type ThreadCreateResponse = {
  thread_id?: unknown;
  threadId?: unknown;
};

function createLangGraphHeaders() {
  const { apiKey, authHeader, authScheme } = getLangGraphEnv();
  const headers = new Headers({
    Accept: "application/json",
    "Content-Type": "application/json",
  });

  if (apiKey) {
    const value =
      authHeader.toLowerCase() === "authorization"
        ? `${authScheme} ${apiKey}`
        : apiKey;

    headers.set(authHeader, value);
  }

  return headers;
}

export async function createThread() {
  const payload = await langGraphFetch<ThreadCreateResponse>("/threads", {
    method: "POST",
    body: JSON.stringify({}),
  });

  const threadId =
    typeof payload.thread_id === "string"
      ? payload.thread_id
      : typeof payload.threadId === "string"
        ? payload.threadId
        : null;

  if (!threadId) {
    throw new RouteError(
      502,
      "LANGGRAPH_INVALID_RESPONSE",
      "LangGraph did not return a thread ID.",
    );
  }

  return threadId;
}

export async function waitForMessageRun(threadId: string, message: string) {
  const { assistantId } = getLangGraphEnv();

  await langGraphFetch(`/threads/${threadId}/runs/wait`, {
    method: "POST",
    body: JSON.stringify({
      assistant_id: assistantId,
      input: {
        messages: [
          {
            role: "user",
            content: message,
          },
        ],
      },
      multitask_strategy: "reject",
    }),
  });

  return fetchThreadState(threadId);
}

export async function waitForResumeRun(threadId: string, answer: string) {
  const { assistantId } = getLangGraphEnv();

  await langGraphFetch(`/threads/${threadId}/runs/wait`, {
    method: "POST",
    body: JSON.stringify({
      assistant_id: assistantId,
      command: {
        resume: answer,
      },
      multitask_strategy: "reject",
    }),
  });

  return fetchThreadState(threadId);
}

export async function fetchThreadState(threadId: string) {
  return langGraphFetch(`/threads/${threadId}/state`, {
    method: "GET",
  });
}

async function langGraphFetch<T = unknown>(path: string, init: RequestInit) {
  const { baseUrl, timeoutMs } = getLangGraphEnv();

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: createLangGraphHeaders(),
      signal: AbortSignal.timeout(timeoutMs),
      cache: "no-store",
    });

    if (!response.ok) {
      const bodyText = await response.text().catch(() => "");
      throw mapLangGraphError(response.status, bodyText);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof RouteError) {
      throw error;
    }

    if (error instanceof Error && error.name === "TimeoutError") {
      throw new RouteError(
        504,
        "LANGGRAPH_TIMEOUT",
        "The research backend took too long to answer. Please try again.",
      );
    }

    throw new RouteError(
      502,
      "LANGGRAPH_UNAVAILABLE",
      "The research backend is currently unavailable.",
    );
  }
}

function mapLangGraphError(status: number, bodyText: string) {
  if (status === 404) {
    return new RouteError(
      404,
      "THREAD_NOT_FOUND",
      "This thread could not be found. Start a new chat to continue.",
    );
  }

  if (status === 409) {
    return new RouteError(
      409,
      "THREAD_BUSY",
      "A previous run is still active on this thread. Wait a moment and retry.",
    );
  }

  if (status === 400 || status === 422) {
    return new RouteError(
      400,
      "LANGGRAPH_BAD_REQUEST",
      bodyText || "The request payload was rejected by the research backend.",
    );
  }

  return new RouteError(
    502,
    "LANGGRAPH_REQUEST_FAILED",
    bodyText || "The research backend returned an unexpected error.",
  );
}

