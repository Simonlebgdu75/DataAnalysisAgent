export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt?: string | null;
  pending?: boolean;
};

export type ShortlistItem = {
  id: string;
  companyName: string;
  description: string | null;
  linkedinSlug: string | null;
  website: string | null;
  countryCode: string | null;
  linkedinEmployees: number | null;
  isPeBacked: boolean | null;
  productsServices: string[];
  formattedRevenue: string | null;
  rerankReason: string | null;
};

export type ApiError = {
  code: string;
  message: string;
};

export type PeQaStateResponse = {
  threadId: string;
  messages: ChatMessage[];
  shortlist: ShortlistItem[];
  errors: ApiError[];
  pendingQuestion?: string;
};

export type PeQaRunResponse =
  | {
      status: "completed";
      threadId: string;
      messages: ChatMessage[];
      shortlist: ShortlistItem[];
      errors: ApiError[];
    }
  | {
      status: "needs_input";
      threadId: string;
      question: string;
      messages: ChatMessage[];
      shortlist: ShortlistItem[];
      errors: ApiError[];
    };

