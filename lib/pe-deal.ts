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

type JsonRecord = Record<string, unknown>;

export function toDealResearchRunResponse(
  threadId: string,
  rawState: unknown,
): DealResearchRunResponse {
  const state = toDealResearchStateResponse(threadId, rawState);

  if (state.pendingQuestion) {
    return {
      status: "needs_input",
      threadId,
      question: state.pendingQuestion,
      messages: state.messages,
      shortlist: state.shortlist,
      targets: state.targets,
      deals: state.deals,
      acquirers: state.acquirers,
      errors: state.errors,
      progress: state.progress,
    };
  }

  return {
    status: "completed",
    threadId,
    messages: state.messages,
    shortlist: state.shortlist,
    targets: state.targets,
    deals: state.deals,
    acquirers: state.acquirers,
    errors: state.errors,
    progress: state.progress,
  };
}

export function toDealResearchStateResponse(
  threadId: string,
  rawState: unknown,
): DealResearchStateResponse {
  const state = asRecord(rawState);
  const values = extractValues(state.values);
  const pendingQuestion = extractPendingQuestion(state);

  return {
    threadId,
    messages: normalizeMessages(values.messages),
    shortlist: normalizeEntities(values.shortlist, "unknown"),
    targets: normalizeEntities(values.target_shortlist, "target"),
    deals: normalizeDeals(values.deals_shortlist),
    acquirers: normalizeEntities(values.acquirers_shortlist, "buyer"),
    errors: normalizeErrors(values.errors),
    progress: normalizeProgress(values.progress),
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
        firstScalarString(record.id) ??
        `${role}-${index}-${text.slice(0, 12)}`,
      role,
      text,
      createdAt:
        firstString(record.created_at, record.createdAt) ?? null,
    });
  });

  return normalized;
}

function normalizeEntities(
  input: unknown,
  fallbackType: DealResearchEntity["entityType"],
): DealResearchEntity[] {
  if (!Array.isArray(input)) {
    return [];
  }

  return input
    .map((item, index) => normalizeEntity(item, index, fallbackType))
    .filter((item): item is DealResearchEntity => Boolean(item));
}

function normalizeEntity(
  input: unknown,
  index: number,
  fallbackType: DealResearchEntity["entityType"],
): DealResearchEntity | null {
  const record = asRecord(input);
  const entityKey = firstScalarString(record.entity_key, record.entityKey);
  const rawType = firstString(record.entity_type, record.entityType);
  const entityType = normalizeEntityType(rawType, record, fallbackType);
  const companyName =
    firstString(
      record.company_name,
      record.linkedin_company_name,
      record.acquirer_name,
      record.name,
    ) ??
    firstScalarString(record.linkedin_slug, record.acquirer_linkedin_slug) ??
    `${entityType === "buyer" ? "Buyer" : "Company"} ${index + 1}`;
  const linkedinSlug = firstString(record.linkedin_slug, record.acquirer_linkedin_slug);
  const sourceDeals = normalizeDeals(record.source_deals ?? record.deals);
  const buildUp = normalizeDeals(record.build_up);
  const shareholding = normalizeShareholdings(
    record.shareholding ?? record.shareholding_targets,
  );

  return {
    id:
      entityKey ??
      firstScalarString(record.id, record.fund_id, linkedinSlug) ??
      `${companyName}-${index}`,
    entityKey,
    entityType,
    entityRoles: normalizeEntityRoles(record.entity_roles, entityType),
    companyName,
    acquirerName: firstString(record.acquirer_name),
    linkedinSlug,
    website: firstString(record.linkedin_website, record.website, record.domain),
    headquarters: firstString(record.linkedin_headquarters, record.formatted_locations),
    countryCodes: firstStringArray(record.country_codes),
    linkedinEmployees: firstNumber(record.linkedin_employees, record.employees),
    description: firstString(
      record.shortlist_rationale,
      record.description,
      record.one_liner,
      record.target_rerank_reason,
      record.rerank_reason,
    ),
    oneLiner: firstString(record.one_liner),
    finalScore: firstNumber(record.final_score, record.acquirer_score, record.rrf_score),
    bestTargetRank: firstNumber(record.best_target_rank, record.target_rank),
    sourceDealCount: firstNumber(record.source_deal_count, record.deal_count),
    buildUpCount: firstNumber(record.build_up_count),
    shareholdingCount: firstNumber(
      record.shareholding_count,
      record.current_shareholding_count,
    ),
    isUnderLbo: firstBoolean(record.is_under_lbo),
    hasSectorBuildUp: firstBoolean(record.has_sector_build_up),
    shortlistRationale: firstString(record.shortlist_rationale),
    rerankReason: firstString(record.rerank_reason, record.target_rerank_reason),
    sourceDeals,
    buildUp,
    shareholding,
  };
}

function normalizeDeals(input: unknown): DealResearchDeal[] {
  if (!Array.isArray(input)) {
    return [];
  }

  return input
    .map((item, index) => {
      const record = asRecord(item);
      const companyName =
        firstString(record.company_name, record.target_company_name) ??
        firstScalarString(record.linkedin_slug) ??
        `Deal ${index + 1}`;
      const id =
        firstScalarString(record.id, record.deal_key, record.synthetic_id) ??
        `${companyName}-${index}`;

      return {
        id,
        companyName,
        linkedinSlug: firstString(record.linkedin_slug),
        acquirerName: firstString(record.acquirer_name),
        fundId: firstScalarString(record.fund_id),
        dealType: firstString(record.deal_type),
        dealYear: firstNumber(record.deal_year),
        country: firstString(record.country),
        description: firstString(record.description),
        sourceType: firstString(record.source_type),
        sourceUrl: firstString(record.source_url),
        isSynthetic: firstBoolean(record.is_synthetic) ?? false,
        relationshipType: firstString(record.relationship_type),
        targetRank: firstNumber(record.target_rank),
        targetCompanyName: firstString(record.target_company_name),
        rerankReason: firstString(record.target_rerank_reason, record.rerank_reason),
      };
    })
    .filter((item) => item.companyName.length > 0);
}

function normalizeShareholdings(input: unknown): DealResearchShareholding[] {
  if (!Array.isArray(input)) {
    return [];
  }

  return input
    .map((item, index) => {
      const record = asRecord(item);
      const companyName =
        firstString(record.company_name, record.acquirer_name) ??
        firstScalarString(record.linkedin_slug) ??
        `Shareholding ${index + 1}`;
      const relationshipType = firstString(record.relationship_type);
      const entryYear = firstNumber(record.entry_year);
      const exitYear = firstNumber(record.exit_year);

      return {
        id:
          firstScalarString(record.id, record.fund_id, record.linkedin_slug) ??
          `${companyName}-${relationshipType ?? "relationship"}-${entryYear ?? index}`,
        companyName,
        acquirerName: firstString(record.acquirer_name),
        linkedinSlug: firstString(record.linkedin_slug),
        relationshipType,
        entryYear,
        exitYear,
        description: firstString(record.description, record.one_liner),
      };
    })
    .filter((item) => item.companyName.length > 0);
}

function normalizeProgress(input: unknown): DealResearchProgress | null {
  const record = asRecord(input);
  if (Object.keys(record).length === 0) {
    return null;
  }

  return {
    step: firstString(record.step),
    status: firstString(record.status),
    mode: firstString(record.mode),
    message: firstString(record.message),
    targetCount: firstNumber(record.target_count, record.targetCount),
    finalShortlistCount: firstNumber(
      record.final_shortlist_count,
      record.finalShortlistCount,
    ),
    dealCount: firstNumber(record.deal_count, record.dealCount),
    buyerCount: firstNumber(record.buyer_count, record.buyerCount),
  };
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

  const clarificationQuestions = record.clarification_questions;
  if (Array.isArray(clarificationQuestions)) {
    const parts = clarificationQuestions
      .map((item, index) => {
        const questionRecord = asRecord(item);
        const question = firstString(questionRecord.question);
        if (!question) {
          return "";
        }
        const options = firstStringArray(questionRecord.options)
          .map((option) => `- ${option}`)
          .join("\n");
        return `${index + 1}. ${question}${options ? `\n${options}` : ""}`;
      })
      .filter(Boolean);

    if (parts.length > 0) {
      return parts.join("\n\n");
    }
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

function normalizeEntityType(
  rawType: string | null,
  record: JsonRecord,
  fallbackType: DealResearchEntity["entityType"],
): DealResearchEntity["entityType"] {
  const value = rawType?.toLowerCase();
  if (value === "buyer" || value === "target") {
    return value;
  }

  if (fallbackType !== "unknown") {
    return fallbackType;
  }

  if (record.acquirer_name || record.fund_id) {
    return "buyer";
  }

  return "target";
}

function normalizeEntityRoles(
  input: unknown,
  entityType: DealResearchEntity["entityType"],
) {
  const roles = firstStringArray(input);
  if (roles.length > 0) {
    return roles;
  }

  return entityType === "unknown" ? [] : [entityType];
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return null;
}

function firstScalarString(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }

    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
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
      const normalized = Number.parseFloat(value);
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

    if (typeof value === "string" && value.trim()) {
      return value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    }
  }

  return [];
}

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" ? (value as JsonRecord) : {};
}
