import { assertAuthenticatedRequest } from "@/lib/auth";
import { getSecurityEnv } from "@/lib/env";
import { createThread, waitForMessageRun } from "@/lib/langgraph";
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

type MessageRequestBody = {
  threadId?: unknown;
  message?: unknown;
};

export async function POST(request: Request) {
  try {
    assertAuthenticatedRequest(request);
    assertSameOrigin(request);

    await enforceRateLimit(getSecurityEnv().messageRateLimitId, request);

    const body = await parseJsonBody<MessageRequestBody>(request);
    const message = typeof body.message === "string" ? body.message.trim() : "";
    const existingThreadId =
      typeof body.threadId === "string" && body.threadId.trim()
        ? body.threadId.trim()
        : undefined;

    if (!message) {
      throw new RouteError(400, "INVALID_INPUT", "Message is required.");
    }

    const threadId = existingThreadId ?? (await createThread());
    const state = await waitForMessageRun(threadId, message);

    return jsonNoStore(toPeQaRunResponse(threadId, state));
  } catch (error) {
    return toErrorResponse(error);
  }
}

