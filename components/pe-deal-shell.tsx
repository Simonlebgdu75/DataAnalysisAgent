"use client";

import { startTransition, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { ChatMarkdown } from "@/components/chat-markdown";
import { TruncatedText } from "@/components/truncated-text";
import type {
  ApiError,
  ChatMessage,
  DealResearchDeal,
  DealResearchEntity,
  DealResearchProgress,
  DealResearchRunResponse,
  DealResearchShareholding,
  DealResearchStateResponse,
} from "@/lib/contracts";

type ApiErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
  };
};

type ResultView = "shortlist" | "buyers" | "targets" | "deals";

type ConversationSnapshot = {
  threadId: string | null;
  messages: ChatMessage[];
  shortlist: DealResearchEntity[];
  targets: DealResearchEntity[];
  deals: DealResearchDeal[];
  acquirers: DealResearchEntity[];
  errors: ApiError[];
  progress: DealResearchProgress | null;
  pendingQuestion: string | null;
};

const DEFAULT_ERROR: ApiError = {
  code: "UNKNOWN_ERROR",
  message: "Something went wrong. Please try again.",
};

const EMPTY_COPY: Record<ResultView, string> = {
  shortlist: "No final entities yet.",
  buyers: "No buyers yet.",
  targets: "No comparable targets yet.",
  deals: "No source deals yet.",
};

function toWebsiteHref(website: string | null) {
  if (!website) {
    return null;
  }

  return website.startsWith("http://") || website.startsWith("https://")
    ? website
    : `https://${website}`;
}

function formatEntityMeta(entity: DealResearchEntity) {
  return (
    [
      entity.countryCodes.length > 0 ? entity.countryCodes.join(", ") : null,
      entity.headquarters,
      entity.linkedinEmployees ? `${entity.linkedinEmployees} employees` : null,
    ]
      .filter(Boolean)
      .join(" · ") || "Not specified"
  );
}

function formatRole(entity: DealResearchEntity) {
  if (entity.entityRoles.length > 0) {
    return entity.entityRoles.map(formatTitle).join(" · ");
  }

  return formatTitle(entity.entityType);
}

function formatTitle(value: string) {
  return value
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function formatDealMeta(deal: DealResearchDeal) {
  return (
    [
      deal.dealType ? formatTitle(deal.dealType) : null,
      deal.dealYear ? String(deal.dealYear) : null,
      deal.isSynthetic ? "Shareholding" : deal.sourceType,
    ]
      .filter(Boolean)
      .join(" · ") || "Deal"
  );
}

function formatShareholdingMeta(item: DealResearchShareholding) {
  return (
    [
      item.relationshipType ? formatTitle(item.relationshipType) : null,
      item.entryYear ? `Entry ${item.entryYear}` : null,
      item.exitYear ? `Exit ${item.exitYear}` : null,
    ]
      .filter(Boolean)
      .join(" · ") || "Shareholding"
  );
}

function resultCountLabel(count: number) {
  return `${count} ${count === 1 ? "row" : "rows"}`;
}

export function PeDealShell() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlThreadId = searchParams.get("thread")?.trim() ?? null;

  const [threadId, setThreadId] = useState<string | null>(urlThreadId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [shortlist, setShortlist] = useState<DealResearchEntity[]>([]);
  const [targets, setTargets] = useState<DealResearchEntity[]>([]);
  const [deals, setDeals] = useState<DealResearchDeal[]>([]);
  const [acquirers, setAcquirers] = useState<DealResearchEntity[]>([]);
  const [errors, setErrors] = useState<ApiError[]>([]);
  const [progress, setProgress] = useState<DealResearchProgress | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [activeView, setActiveView] = useState<ResultView>("shortlist");
  const [isWorking, setIsWorking] = useState(false);
  const [isHydrating, setIsHydrating] = useState(false);

  const skipNextThreadSync = useRef<string | null>(null);
  const activeRequestId = useRef(0);

  const tabs: Array<{ id: ResultView; label: string; count: number }> = [
    { id: "shortlist", label: "Final", count: shortlist.length },
    { id: "buyers", label: "Buyers", count: acquirers.length },
    { id: "targets", label: "Targets", count: targets.length },
    { id: "deals", label: "Deals", count: deals.length },
  ];

  function buildSnapshot(): ConversationSnapshot {
    return {
      threadId,
      messages,
      shortlist,
      targets,
      deals,
      acquirers,
      errors,
      progress,
      pendingQuestion,
    };
  }

  function applyState(payload: DealResearchStateResponse | DealResearchRunResponse) {
    setThreadId(payload.threadId);
    setMessages(payload.messages);
    setShortlist(payload.shortlist);
    setTargets(payload.targets);
    setDeals(payload.deals);
    setAcquirers(payload.acquirers);
    setErrors(payload.errors);
    setProgress(payload.progress);

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
    setTargets([]);
    setDeals([]);
    setAcquirers([]);
    setErrors([]);
    setProgress(null);
    setPendingQuestion(null);
    setDraft("");
    setActiveView("shortlist");
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
          `/api/pe-deal/state?threadId=${encodeURIComponent(urlThreadId)}`,
          {
            cache: "no-store",
          },
        );

        const payload = (await response.json().catch(() => null)) as
          | DealResearchStateResponse
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

        applyState(payload as DealResearchStateResponse);
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

    const route = pendingQuestion ? "/api/pe-deal/resume" : "/api/pe-deal/message";
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
        | DealResearchRunResponse
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
      setTargets(snapshot.targets);
      setDeals(snapshot.deals);
      setAcquirers(snapshot.acquirers);
      setErrors(snapshot.errors);
      setProgress(snapshot.progress);
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
              <h2>Deal research</h2>
              <p>Buyer discovery from comparable targets and transaction evidence.</p>
            </div>
            <div className="panel-actions">
              <span className="status-chip" data-active={isWorking || isHydrating}>
                {isHydrating
                  ? "Restoring thread..."
                  : isWorking
                    ? "Researching..."
                    : progress?.step
                      ? formatTitle(progress.step)
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
                {pendingQuestion ? "Clarification" : "pe_deal"}
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
                <h3>Start a buyer search</h3>
                <p>
                  Describe the company or market. The graph will clarify the buyer
                  logic before retrieving comparable targets, deals, and acquirers.
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
                  ? "Answer the clarification question..."
                  : "Describe the target company or buyer thesis..."
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
              <h3>Results</h3>
              <p>{progress?.message ?? "Deal research output."}</p>
            </div>
          </div>

          <div className="result-tabs" role="tablist" aria-label="Deal research results">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className="result-tab"
                data-active={activeView === tab.id}
                type="button"
                role="tab"
                aria-selected={activeView === tab.id}
                onClick={() => setActiveView(tab.id)}
              >
                <span>{tab.label}</span>
                <span>{tab.count}</span>
              </button>
            ))}
          </div>

          <div className="metrics-strip" aria-label="Result counts">
            <span>{resultCountLabel(shortlist.length)} final</span>
            <span>{resultCountLabel(acquirers.length)} buyers</span>
            <span>{resultCountLabel(targets.length)} targets</span>
            <span>{resultCountLabel(deals.length)} deals</span>
          </div>

          <div className="shortlist-scroll">
            {activeView === "shortlist" ? (
              <EntityTable
                emptyCopy={EMPTY_COPY.shortlist}
                entities={shortlist}
              />
            ) : null}
            {activeView === "buyers" ? (
              <EntityTable emptyCopy={EMPTY_COPY.buyers} entities={acquirers} />
            ) : null}
            {activeView === "targets" ? (
              <EntityTable emptyCopy={EMPTY_COPY.targets} entities={targets} />
            ) : null}
            {activeView === "deals" ? (
              <DealTable deals={deals} emptyCopy={EMPTY_COPY.deals} />
            ) : null}
          </div>
        </aside>
      </section>
    </main>
  );
}

function EntityTable({
  emptyCopy,
  entities,
}: {
  emptyCopy: string;
  entities: DealResearchEntity[];
}) {
  if (entities.length === 0) {
    return <div className="shortlist-empty">{emptyCopy}</div>;
  }

  return (
    <table className="shortlist-table result-table">
      <thead>
        <tr>
          <th>Entity</th>
          <th>Profile</th>
          <th>Evidence</th>
        </tr>
      </thead>
      <tbody>
        {entities.map((entity) => (
          <tr key={entity.id}>
            <td>
              <div className="company-cell">
                <span className="company-name">{entity.companyName}</span>
                <span className="entity-type">{formatRole(entity)}</span>
                {entity.website ? (
                  <a
                    className="company-link"
                    href={toWebsiteHref(entity.website) ?? undefined}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {entity.website}
                  </a>
                ) : null}
                <EntityBadges entity={entity} />
              </div>
            </td>
            <td>
              <TruncatedText
                className="footprint-text"
                limit={150}
                text={formatEntityMeta(entity)}
              />
              {entity.description ? (
                <TruncatedText
                  className="rationale-text"
                  limit={220}
                  text={entity.description}
                />
              ) : null}
            </td>
            <td>
              <EvidenceList entity={entity} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EntityBadges({ entity }: { entity: DealResearchEntity }) {
  const badges = [
    entity.sourceDealCount ? `${entity.sourceDealCount} source deals` : null,
    entity.buildUpCount ? `${entity.buildUpCount} build-up` : null,
    entity.shareholdingCount ? `${entity.shareholdingCount} holdings` : null,
    entity.isUnderLbo ? "LBO signal" : null,
    entity.hasSectorBuildUp ? "Sector build-up" : null,
  ].filter(Boolean);

  if (badges.length === 0) {
    return null;
  }

  return (
    <div className="products-list">
      {badges.map((badge) => (
        <span key={badge}>{badge}</span>
      ))}
    </div>
  );
}

function EvidenceList({ entity }: { entity: DealResearchEntity }) {
  const deals = entity.sourceDeals.length > 0 ? entity.sourceDeals : entity.buildUp;
  const holdings = entity.shareholding;

  if (deals.length === 0 && holdings.length === 0) {
    return (
      <TruncatedText
        className="footprint-text"
        limit={220}
        text={entity.rerankReason ?? entity.shortlistRationale ?? "No evidence attached."}
      />
    );
  }

  return (
    <div className="relation-list">
      {deals.slice(0, 3).map((deal) => (
        <div className="relation-item" key={deal.id}>
          <span>{deal.companyName}</span>
          <small>
            {deal.acquirerName ? `${deal.acquirerName} · ` : ""}
            {formatDealMeta(deal)}
          </small>
        </div>
      ))}
      {holdings.slice(0, 2).map((holding) => (
        <div className="relation-item" key={holding.id}>
          <span>{holding.companyName}</span>
          <small>{formatShareholdingMeta(holding)}</small>
        </div>
      ))}
    </div>
  );
}

function DealTable({
  deals,
  emptyCopy,
}: {
  deals: DealResearchDeal[];
  emptyCopy: string;
}) {
  if (deals.length === 0) {
    return <div className="shortlist-empty">{emptyCopy}</div>;
  }

  return (
    <table className="shortlist-table result-table">
      <thead>
        <tr>
          <th>Target</th>
          <th>Acquirer</th>
          <th>Deal</th>
        </tr>
      </thead>
      <tbody>
        {deals.map((deal) => (
          <tr key={deal.id}>
            <td>
              <div className="company-cell">
                <span className="company-name">{deal.companyName}</span>
                {deal.rerankReason ? (
                  <TruncatedText
                    className="company-meta"
                    limit={160}
                    text={deal.rerankReason}
                  />
                ) : null}
              </div>
            </td>
            <td>
              <div className="footprint-text">
                {deal.acquirerName ?? "Not specified"}
              </div>
            </td>
            <td>
              <div className="company-cell">
                <span className="footprint-text">{formatDealMeta(deal)}</span>
                {deal.description ? (
                  <TruncatedText
                    className="rationale-text"
                    limit={180}
                    text={deal.description}
                  />
                ) : null}
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
