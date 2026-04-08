import type {
  ApiError,
  ChatMessage,
  PeQaRunResponse,
  PeQaStateResponse,
  ShortlistItem,
} from "@/lib/contracts";

type JsonRecord = Record<string, unknown>;

export function toPeQaRunResponse(
  threadId: string,
  rawState: unknown,
): PeQaRunResponse {
  const state = toPeQaStateResponse(threadId, rawState);

  if (state.pendingQuestion) {
    return {
      status: "needs_input",
      threadId,
      question: state.pendingQuestion,
      messages: state.messages,
      shortlist: state.shortlist,
      errors: state.errors,
    };
  }

  return {
    status: "completed",
    threadId,
    messages: state.messages,
    shortlist: state.shortlist,
    errors: state.errors,
  };
}

export function toPeQaStateResponse(
  threadId: string,
  rawState: unknown,
): PeQaStateResponse {
  const state = asRecord(rawState);
  const values = extractValues(state.values);

  const pendingQuestion = extractPendingQuestion(state);

  return {
    threadId,
    messages: normalizeMessages(values.messages),
    shortlist: normalizeShortlist(values.shortlist),
    errors: normalizeErrors(values.errors),
    ...(pendingQuestion ? { pendingQuestion } : {}),
  };
}

function normalizeMessages(input: unknown): ChatMessage[] {
  if (!Array.isArray(input)) {
    return [];
  }

  const normalized: ChatMessage[] = [];

  input.forEach((item, index) => {
    const record = asRecord(item);
    const role = normalizeRole(record.role ?? record.type);
    if (!role) {
      return;
    }

    const text = extractText(record.content);
    if (!text.trim()) {
      return;
    }

    normalized.push({
      id:
        typeof record.id === "string"
          ? record.id
          : `${role}-${index}-${text.slice(0, 12)}`,
      role,
      text,
      createdAt:
        typeof record.created_at === "string"
          ? record.created_at
          : typeof record.createdAt === "string"
            ? record.createdAt
            : null,
    });
  });

  return normalized;
}

function normalizeShortlist(input: unknown): ShortlistItem[] {
  if (!Array.isArray(input)) {
    return [];
  }

  return input
    .map((item, index) => {
      const record = asRecord(item);
      const linkedinSlug = firstString(record.linkedin_slug, record.linkedinSlug);
      const companyName =
        firstString(record.company_name, record.companyName, record.name) ??
        linkedinSlug ??
        `Company ${index + 1}`;

      return {
        id:
          firstString(record.id, record.linkedin_slug, record.linkedinSlug) ??
          `${companyName}-${index}`,
        companyName,
        description: firstString(
          record.description,
          record.tagline,
          record.summary,
        ),
        linkedinSlug,
        website: firstString(record.website, record.domain),
        countryCode: firstString(record.country_code, record.countryCode),
        linkedinEmployees: firstNumber(
          record.linkedin_employees,
          record.linkedinEmployees,
          record.employees,
        ),
        isPeBacked: firstBoolean(record.is_pe_backed, record.isPeBacked),
        productsServices: firstStringArray(
          record.products_services,
          record.productsServices,
        ),
        formattedRevenue: firstString(
          record.formatted_revenue,
          record.formattedRevenue,
          record.revenue,
        ),
        rerankReason: firstString(record.rerank_reason, record.rerankReason),
      };
    })
    .filter((item) => item.companyName.length > 0);
}

function normalizeErrors(input: unknown): ApiError[] {
  if (Array.isArray(input)) {
    return input
      .map((item, index) => normalizeSingleError(item, index))
      .filter(Boolean) as ApiError[];
  }

  if (input === undefined || input === null) {
    return [];
  }

  const single = normalizeSingleError(input, 0);
  return single ? [single] : [];
}

function normalizeSingleError(input: unknown, index: number): ApiError | null {
  if (typeof input === "string" && input.trim()) {
    return {
      code: `ERROR_${index + 1}`,
      message: input.trim(),
    };
  }

  const record = asRecord(input);
  const message = firstString(record.message, record.error, record.detail);
  if (!message) {
    return null;
  }

  return {
    code: firstString(record.code, record.type) ?? `ERROR_${index + 1}`,
    message,
  };
}

function extractPendingQuestion(state: JsonRecord) {
  const interrupts = [
    ...flattenInterrupts(state.interrupts),
    ...flattenTaskInterrupts(state.tasks),
  ];

  for (const interrupt of interrupts) {
    const question = interruptValueToQuestion(interrupt.value);
    if (question) {
      return question;
    }
  }

  return undefined;
}

function flattenTaskInterrupts(tasks: unknown) {
  if (!Array.isArray(tasks)) {
    return [];
  }

  return tasks.flatMap((task) => flattenInterrupts(asRecord(task).interrupts));
}

function flattenInterrupts(input: unknown): JsonRecord[] {
  if (Array.isArray(input)) {
    return input.map(asRecord);
  }

  const record = asRecord(input);
  return Object.values(record).flatMap((value) => {
    if (Array.isArray(value)) {
      return value.map(asRecord);
    }

    if (value && typeof value === "object") {
      return [asRecord(value)];
    }

    return [];
  });
}

function interruptValueToQuestion(input: unknown): string | undefined {
  if (typeof input === "string" && input.trim()) {
    return input.trim();
  }

  if (Array.isArray(input)) {
    const parts = input
      .map((item) => interruptValueToQuestion(item))
      .filter((item): item is string => Boolean(item));

    return parts.length > 0 ? parts.join("\n\n") : undefined;
  }

  const record = asRecord(input);

  const direct = firstString(
    record.question,
    record.prompt,
    record.message,
    record.text,
  );
  if (direct) {
    return direct;
  }

  const fromQuestions = firstStringArray(record.questions);
  if (fromQuestions.length > 0) {
    return fromQuestions.join("\n\n");
  }

  if ("value" in record) {
    return interruptValueToQuestion(record.value);
  }

  return undefined;
}

function extractValues(input: unknown): JsonRecord {
  if (Array.isArray(input)) {
    return input.length > 0 ? asRecord(input[0]) : {};
  }

  return asRecord(input);
}

function extractText(input: unknown): string {
  if (typeof input === "string") {
    return input;
  }

  if (Array.isArray(input)) {
    return input
      .map((item) => extractText(item))
      .filter(Boolean)
      .join("\n\n")
      .trim();
  }

  const record = asRecord(input);

  const direct = firstString(record.text, record.value);
  if (direct) {
    return direct;
  }

  if (record.type === "text") {
    return firstString(record.text, record.value) ?? "";
  }

  if ("content" in record) {
    return extractText(record.content);
  }

  return "";
}

function normalizeRole(input: unknown): ChatMessage["role"] | null {
  if (input === "human" || input === "user") {
    return "user";
  }

  if (input === "ai" || input === "assistant") {
    return "assistant";
  }

  return null;
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return null;
}

function firstNumber(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const normalized = Number.parseInt(value, 10);
      if (Number.isFinite(normalized)) {
        return normalized;
      }
    }
  }

  return null;
}

function firstBoolean(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "boolean") {
      return value;
    }
  }

  return null;
}

function firstStringArray(...values: unknown[]) {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value
        .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
        .map((item) => item.trim());
    }
  }

  return [];
}

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" ? (value as JsonRecord) : {};
}

