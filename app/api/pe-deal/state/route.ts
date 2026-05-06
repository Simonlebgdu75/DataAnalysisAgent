import { assertAuthenticatedRequest } from "@/lib/auth";
import { fetchThreadState } from "@/lib/langgraph";
import { toDealResearchStateResponse } from "@/lib/pe-deal";
import { RouteError, jsonNoStore, toErrorResponse } from "@/lib/request";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    assertAuthenticatedRequest(request);

    const { searchParams } = new URL(request.url);
    const threadId = searchParams.get("threadId")?.trim();

    if (!threadId) {
      throw new RouteError(400, "INVALID_INPUT", "threadId is required.");
    }

    const state = await fetchThreadState(threadId);
    return jsonNoStore(toDealResearchStateResponse(threadId, state));
  } catch (error) {
    return toErrorResponse(error);
  }
}
