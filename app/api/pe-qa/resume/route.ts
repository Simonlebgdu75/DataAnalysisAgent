import { assertAuthenticatedRequest } from "@/lib/auth";
import { getSecurityEnv } from "@/lib/env";
import { waitForResumeRun } from "@/lib/langgraph";
import { toPeQaRunResponse } from "@/lib/pe-qa";
import { enforceRateLimit } from "@/lib/rate-limit";
import {
  RouteError,
  assertSameOrigin,
  jsonNoStore,
  parseJsonBody,
  toErrorResponse,
} from "@/lib/request";

export const runtime = "nodejs";
export const maxDuration = 60;

type ResumeRequestBody = {
  threadId?: unknown;
  answer?: unknown;
};

export async function POST(request: Request) {
  try {
    assertAuthenticatedRequest(request);
    assertSameOrigin(request);

    await enforceRateLimit(getSecurityEnv().resumeRateLimitId, request);

    const body = await parseJsonBody<ResumeRequestBody>(request);
    const threadId =
      typeof body.threadId === "string" ? body.threadId.trim() : "";
    const answer = typeof body.answer === "string" ? body.answer.trim() : "";

    if (!threadId) {
      throw new RouteError(400, "INVALID_INPUT", "Thread ID is required.");
    }

    if (!answer) {
      throw new RouteError(400, "INVALID_INPUT", "Answer is required.");
    }

    const state = await waitForResumeRun(threadId, answer);
    return jsonNoStore(toPeQaRunResponse(threadId, state));
  } catch (error) {
    return toErrorResponse(error);
  }
}

