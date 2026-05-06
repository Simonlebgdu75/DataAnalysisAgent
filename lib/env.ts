type AuthEnv = {
  appGatePasswordHash: string;
  appGateSessionSecret: string;
};

type LangGraphEnv = {
  baseUrl: string;
  apiKey: string | null;
  authHeader: string;
  authScheme: string;
  assistantId: string;
  timeoutMs: number;
};

type SecurityEnv = {
  loginRateLimitId: string | null;
  messageRateLimitId: string | null;
  resumeRateLimitId: string | null;
};

export function getAuthEnv(): AuthEnv {
  return {
    appGatePasswordHash: getRequiredEnv("APP_GATE_PASSWORD_HASH"),
    appGateSessionSecret: getRequiredEnv("APP_GATE_SESSION_SECRET"),
  };
}

export function getLangGraphEnv(): LangGraphEnv {
  const baseUrl = getRequiredEnv("LANGGRAPH_BASE_URL").replace(/\/+$/, "");
  const timeoutMs = parsePositiveInt(process.env.LANGGRAPH_TIMEOUT_MS, 55_000);

  return {
    baseUrl,
    apiKey: getOptionalEnv("LANGGRAPH_API_KEY"),
    authHeader: getOptionalEnv("LANGGRAPH_AUTH_HEADER") ?? "x-api-key",
    authScheme: getOptionalEnv("LANGGRAPH_AUTH_SCHEME") ?? "Bearer",
    assistantId: getOptionalEnv("LANGGRAPH_ASSISTANT_ID") ?? "pe_deal",
    timeoutMs,
  };
}

export function getSecurityEnv(): SecurityEnv {
  return {
    loginRateLimitId: getOptionalEnv("RATE_LIMIT_LOGIN_ID"),
    messageRateLimitId: getOptionalEnv("RATE_LIMIT_MESSAGE_ID"),
    resumeRateLimitId: getOptionalEnv("RATE_LIMIT_RESUME_ID"),
  };
}

function getRequiredEnv(name: string) {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
}

function getOptionalEnv(name: string) {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

function parsePositiveInt(rawValue: string | undefined, fallback: number) {
  if (!rawValue) {
    return fallback;
  }

  const parsed = Number.parseInt(rawValue, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
