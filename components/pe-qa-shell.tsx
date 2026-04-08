"use client";

import { startTransition, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { ChatMarkdown } from "@/components/chat-markdown";
import { TruncatedText } from "@/components/truncated-text";
import type {
  ApiError,
  ChatMessage,
  PeQaRunResponse,
  PeQaStateResponse,
  ShortlistItem,
} from "@/lib/contracts";

type ApiErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
  };
};

type ConversationSnapshot = {
  threadId: string | null;
  messages: ChatMessage[];
  shortlist: ShortlistItem[];
  errors: ApiError[];
  pendingQuestion: string | null;
};

const DEFAULT_ERROR: ApiError = {
  code: "UNKNOWN_ERROR",
  message: "Something went wrong. Please try again.",
};

function toWebsiteHref(website: string | null) {
  if (!website) {
    return null;
  }

  return website.startsWith("http://") || website.startsWith("https://")
    ? website
    : `https://${website}`;
}

function formatFootprint(item: ShortlistItem) {
  return (
    [
      item.countryCode,
      item.linkedinEmployees ? `${item.linkedinEmployees} employees` : null,
      item.isPeBacked === null
        ? null
        : item.isPeBacked
          ? "PE-backed"
          : "Not PE-backed",
    ]
      .filter(Boolean)
      .join(" · ") || "Not specified"
  );
}

export function PeQaShell() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlThreadId = searchParams.get("thread")?.trim() ?? null;

  const [threadId, setThreadId] = useState<string | null>(urlThreadId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [shortlist, setShortlist] = useState<ShortlistItem[]>([]);
  const [errors, setErrors] = useState<ApiError[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [isWorking, setIsWorking] = useState(false);
  const [isHydrating, setIsHydrating] = useState(false);

  const skipNextThreadSync = useRef<string | null>(null);
  const activeRequestId = useRef(0);

  function buildSnapshot(): ConversationSnapshot {
    return {
      threadId,
      messages,
      shortlist,
      errors,
      pendingQuestion,
    };
  }

  function applyState(payload: PeQaStateResponse | PeQaRunResponse) {
    setThreadId(payload.threadId);
    setMessages(payload.messages);
    setShortlist(payload.shortlist);
    setErrors(payload.errors);
    if ("status" in payload) {
      setPendingQuestion(
        payload.status === "needs_input" ? payload.question : null,
      );
      return;
    }

    setPendingQuestion(payload.pendingQuestion ?? null);
  }

  function syncThreadInUrl(nextThreadId: string | null) {
    skipNextThreadSync.current = nextThreadId;

    startTransition(() => {
      const destination = nextThreadId
        ? `/app?thread=${encodeURIComponent(nextThreadId)}`
        : "/app";
      router.replace(destination, { scroll: false });
    });
  }

  function resetConversation() {
    setThreadId(null);
    setMessages([]);
    setShortlist([]);
    setErrors([]);
    setPendingQuestion(null);
    setDraft("");
  }

  useEffect(() => {
    if (!urlThreadId) {
      if (skipNextThreadSync.current === null) {
        resetConversation();
      } else {
        skipNextThreadSync.current = null;
      }
      return;
    }

    if (skipNextThreadSync.current === urlThreadId) {
      skipNextThreadSync.current = null;
      return;
    }

    const requestId = activeRequestId.current + 1;
    activeRequestId.current = requestId;
    setIsHydrating(true);
    setErrors([]);

    void (async () => {
      try {
        const response = await fetch(
          `/api/pe-qa/state?threadId=${encodeURIComponent(urlThreadId)}`,
          {
            cache: "no-store",
          },
        );

        const payload = (await response.json().catch(() => null)) as
          | PeQaStateResponse
          | ApiErrorEnvelope
          | null;

        if (!response.ok) {
          throw new Error(
            payload && "error" in payload
              ? payload.error?.message ?? "Unable to restore this thread."
              : "Unable to restore this thread.",
          );
        }

        if (activeRequestId.current !== requestId || payload === null) {
          return;
        }

        applyState(payload as PeQaStateResponse);
      } catch (loadError) {
        if (activeRequestId.current !== requestId) {
          return;
        }

        resetConversation();
        setErrors([
          {
            code: "THREAD_RESTORE_FAILED",
            message:
              loadError instanceof Error
                ? loadError.message
                : "Unable to restore this thread.",
          },
        ]);
      } finally {
        if (activeRequestId.current === requestId) {
          setIsHydrating(false);
        }
      }
    })();
  }, [urlThreadId]);

  async function submitConversation(event?: React.FormEvent<HTMLFormElement>) {
    event?.preventDefault();

    if (isWorking || isHydrating) {
      return;
    }

    const content = draft.trim();
    if (!content) {
      return;
    }

    const snapshot = buildSnapshot();
    const nextUserMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: "user",
      text: content,
      pending: true,
    };

    setDraft("");
    setIsWorking(true);
    setErrors([]);
    setMessages((current) => [...current, nextUserMessage]);

    const route = pendingQuestion ? "/api/pe-qa/resume" : "/api/pe-qa/message";
    const body = pendingQuestion
      ? { threadId, answer: content }
      : { threadId, message: content };

    try {
      const response = await fetch(route, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const payload = (await response.json().catch(() => null)) as
        | PeQaRunResponse
        | ApiErrorEnvelope
        | null;

      if (!response.ok) {
        throw new Error(
          payload && "error" in payload
            ? payload.error?.message ?? "The request failed."
            : "The request failed.",
        );
      }

      if (!payload || !("threadId" in payload)) {
        throw new Error("The backend returned an invalid payload.");
      }

      applyState(payload);
      syncThreadInUrl(payload.threadId);
    } catch (submitError) {
      setThreadId(snapshot.threadId);
      setMessages(snapshot.messages);
      setShortlist(snapshot.shortlist);
      setPendingQuestion(snapshot.pendingQuestion);
      setDraft(content);
      setErrors([
        {
          code: "REQUEST_FAILED",
          message:
            submitError instanceof Error
              ? submitError.message
              : DEFAULT_ERROR.message,
        },
      ]);
    } finally {
      setIsWorking(false);
    }
  }

  async function handleLogout() {
    if (isWorking || isHydrating) {
      return;
    }

    setIsWorking(true);

    try {
      await fetch("/api/auth/logout", {
        method: "POST",
      });
    } finally {
      startTransition(() => {
        router.replace("/login");
      });
    }
  }

  function handleNewChat() {
    if (isWorking) {
      return;
    }

    resetConversation();
    syncThreadInUrl(null);
  }

  return (
    <main className="shell">
      <section className="app-grid">
        <section className="panel conversation-panel">
          <div className="panel-header">
            <div>
              <h2>Conversation</h2>
              <p>
                First turn may interrupt for clarification before search starts.
              </p>
            </div>
            <div className="panel-actions">
              <span className="status-chip" data-active={isWorking || isHydrating}>
                {isHydrating
                  ? "Restoring thread…"
                  : isWorking
                    ? "Recherche en cours…"
                    : "Ready"}
              </span>
              <button
                className="button-secondary"
                onClick={handleNewChat}
                type="button"
              >
                New chat
              </button>
              <button className="button-ghost" onClick={handleLogout} type="button">
                Logout
              </button>
              <span className="helper-pill">
                {pendingQuestion ? "Awaiting clarification" : "Standard turn"}
              </span>
            </div>
          </div>

          <div className="conversation-scroll">
            {errors.length > 0 ? (
              <div className="error-banner" role="alert">
                <h3>Request issue</h3>
                <p>{errors[0]?.message ?? DEFAULT_ERROR.message}</p>
              </div>
            ) : null}

            {messages.length === 0 ? (
              <div className="empty-state">
                <h3>Start a new sourcing conversation</h3>
                <p>
                  Describe a market, competitor set, or investment theme. If the
                  graph needs more context, it will stop with a clarification
                  question before building the shortlist.
                </p>
              </div>
            ) : (
              <div className="message-list">
                {messages.map((message) => (
                  <article
                    key={message.id}
                    className="message"
                    data-role={message.role}
                  >
                    <div className="message-avatar" aria-hidden="true">
                      {message.role === "assistant" ? "AI" : "You"}
                    </div>
                    <div className="message-card">
                      <div className="message-meta">
                        <span>
                          {message.role === "assistant" ? "Assistant" : "User"}
                        </span>
                        {message.pending ? <span className="pending-dot" /> : null}
                      </div>
                      <ChatMarkdown>{message.text}</ChatMarkdown>
                    </div>
                  </article>
                ))}
              </div>
            )}

            {pendingQuestion ? (
              <div className="inline-card clarification-card">
                <h3>Clarification required</h3>
                <p>
                  `pe_qa` suspended on the first-turn context step. Your next
                  answer will call `/api/pe-qa/resume` on the current thread.
                </p>
                <div className="clarification-question">{pendingQuestion}</div>
              </div>
            ) : null}
          </div>

          <form className="composer" onSubmit={submitConversation}>
            <label className="sr-only" htmlFor="composer">
              Message input
            </label>
            <textarea
              id="composer"
              value={draft}
              disabled={isWorking || isHydrating}
              placeholder={
                pendingQuestion
                  ? "Answer the clarification question to resume the thread…"
                  : "Describe the market or company set you want `pe_qa` to investigate…"
              }
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitConversation();
                }
              }}
            />
            <div className="composer-footer">
              <button className="button" disabled={isWorking || isHydrating} type="submit">
                {pendingQuestion ? "Resume thread" : "Send message"}
              </button>
            </div>
          </form>
        </section>

        <aside className="panel side-panel">
          <div className="panel-header">
            <div>
              <h3>Shortlist</h3>
              <p>Companies shortlisted for this request.</p>
            </div>
            <div className="panel-actions">
              <span className="helper-pill">{shortlist.length} rows</span>
            </div>
          </div>

          <div className="shortlist-scroll">
            {shortlist.length === 0 ? (
              <div className="shortlist-empty">
                No shortlist yet. Once `search_companies` completes, the ranked
                companies will appear here without exposing hidden graph state.
              </div>
            ) : (
              <table className="shortlist-table">
                <thead>
                  <tr>
                    <th>Company</th>
                    <th>Footprint</th>
                    <th>Revenue</th>
                  </tr>
                </thead>
                <tbody>
                  {shortlist.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <div className="company-cell">
                          <span className="company-name">{item.companyName}</span>
                          {item.description ? (
                            <TruncatedText
                              className="company-meta"
                              limit={150}
                              text={item.description}
                            />
                          ) : null}
                          {item.website ? (
                            <a
                              className="company-link"
                              href={toWebsiteHref(item.website) ?? undefined}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {item.website}
                            </a>
                          ) : null}
                          {item.productsServices.length > 0 ? (
                            <div className="products-list">
                              {item.productsServices.slice(0, 3).map((product) => (
                                <span key={product}>{product}</span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      </td>
                      <td>
                        <TruncatedText
                          className="footprint-text"
                          limit={150}
                          text={formatFootprint(item)}
                        />
                        {item.rerankReason ? (
                          <div className="rationale-text">{item.rerankReason}</div>
                        ) : null}
                      </td>
                      <td>
                        <div className="footprint-text">
                          {item.formattedRevenue ?? "Not specified"}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
