import { checkRateLimit } from "@vercel/firewall";

import { RouteError } from "@/lib/request";

export async function enforceRateLimit(
  rateLimitId: string | null,
  request: Request,
) {
  if (!rateLimitId) {
    return;
  }

  try {
    const result = await checkRateLimit(rateLimitId, { request });
    if (result.rateLimited) {
      throw new RouteError(
        429,
        "RATE_LIMITED",
        "Too many requests. Wait a moment and try again.",
      );
    }
  } catch (error) {
    if (error instanceof RouteError) {
      throw error;
    }

    console.error("Failed to evaluate rate limit", error);
  }
}

