import { isAuthenticatedRequest } from "@/lib/auth";
import { jsonNoStore, toErrorResponse } from "@/lib/request";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    return jsonNoStore({ authenticated: isAuthenticatedRequest(request) });
  } catch (error) {
    return toErrorResponse(error);
  }
}

