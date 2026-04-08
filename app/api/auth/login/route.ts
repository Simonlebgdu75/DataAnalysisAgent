import { createAuthenticatedSession, verifyAppGatePassword } from "@/lib/auth";
import { getSecurityEnv } from "@/lib/env";
import { enforceRateLimit } from "@/lib/rate-limit";
import {
  RouteError,
  assertSameOrigin,
  jsonNoStore,
  parseJsonBody,
  toErrorResponse,
} from "@/lib/request";

export const runtime = "nodejs";

type LoginRequestBody = {
  password?: unknown;
};

export async function POST(request: Request) {
  try {
    assertSameOrigin(request);

    const body = await parseJsonBody<LoginRequestBody>(request);
    const password = typeof body.password === "string" ? body.password.trim() : "";

    await enforceRateLimit(getSecurityEnv().loginRateLimitId, request);

    if (!password) {
      throw new RouteError(400, "INVALID_INPUT", "Password is required.");
    }

    const isValid = verifyAppGatePassword(password);
    if (!isValid) {
      throw new RouteError(
        401,
        "INVALID_CREDENTIALS",
        "The shared password is invalid.",
      );
    }

    await createAuthenticatedSession();

    return jsonNoStore({ ok: true });
  } catch (error) {
    return toErrorResponse(error);
  }
}

