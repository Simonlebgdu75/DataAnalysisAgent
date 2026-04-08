import { clearAuthenticatedSession } from "@/lib/auth";
import { assertSameOrigin, jsonNoStore, toErrorResponse } from "@/lib/request";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    assertSameOrigin(request);
    await clearAuthenticatedSession();
    return jsonNoStore({ ok: true });
  } catch (error) {
    return toErrorResponse(error);
  }
}

