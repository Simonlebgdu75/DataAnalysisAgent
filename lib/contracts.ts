export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt?: string | null;
  pending?: boolean;
};

export type DealResearchDeal = {
  id: string;
  companyName: string;
  linkedinSlug: string | null;
  acquirerName: string | null;
  fundId: string | null;
  dealType: string | null;
  dealYear: number | null;
  country: string | null;
  description: string | null;
  sourceType: string | null;
  sourceUrl: string | null;
  isSynthetic: boolean;
  relationshipType: string | null;
  targetRank: number | null;
  targetCompanyName: string | null;
  rerankReason: string | null;
};

export type DealResearchShareholding = {
  id: string;
  companyName: string;
  acquirerName: string | null;
  linkedinSlug: string | null;
  relationshipType: string | null;
  entryYear: number | null;
  exitYear: number | null;
  description: string | null;
};

export type DealResearchEntity = {
  id: string;
  entityKey: string | null;
  entityType: "buyer" | "target" | "unknown";
  entityRoles: string[];
  companyName: string;
  acquirerName: string | null;
  linkedinSlug: string | null;
  website: string | null;
  headquarters: string | null;
  countryCodes: string[];
  linkedinEmployees: number | null;
  description: string | null;
  oneLiner: string | null;
  finalScore: number | null;
  bestTargetRank: number | null;
  sourceDealCount: number | null;
  buildUpCount: number | null;
  shareholdingCount: number | null;
  isUnderLbo: boolean | null;
  hasSectorBuildUp: boolean | null;
  shortlistRationale: string | null;
  rerankReason: string | null;
  sourceDeals: DealResearchDeal[];
  buildUp: DealResearchDeal[];
  shareholding: DealResearchShareholding[];
};

export type DealResearchProgress = {
  step: string | null;
  status: string | null;
  mode: string | null;
  message: string | null;
  targetCount: number | null;
  finalShortlistCount: number | null;
  dealCount: number | null;
  buyerCount: number | null;
};

export type ApiError = {
  code: string;
  message: string;
};

export type DealResearchStateResponse = {
  threadId: string;
  messages: ChatMessage[];
  shortlist: DealResearchEntity[];
  targets: DealResearchEntity[];
  deals: DealResearchDeal[];
  acquirers: DealResearchEntity[];
  errors: ApiError[];
  progress: DealResearchProgress | null;
  pendingQuestion?: string;
};

export type DealResearchRunResponse =
  | {
      status: "completed";
      threadId: string;
      messages: ChatMessage[];
      shortlist: DealResearchEntity[];
      targets: DealResearchEntity[];
      deals: DealResearchDeal[];
      acquirers: DealResearchEntity[];
      errors: ApiError[];
      progress: DealResearchProgress | null;
    }
  | {
      status: "needs_input";
      threadId: string;
      question: string;
      messages: ChatMessage[];
      shortlist: DealResearchEntity[];
      targets: DealResearchEntity[];
      deals: DealResearchDeal[];
      acquirers: DealResearchEntity[];
      errors: ApiError[];
      progress: DealResearchProgress | null;
    };
