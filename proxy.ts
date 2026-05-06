import { NextRequest, NextResponse } from "next/server";

import { isAuthenticatedRequest } from "@/lib/auth";

export function proxy(request: NextRequest) {
  if (isAuthenticatedRequest(request)) {
    return NextResponse.next();
  }

  if (request.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json(
      {
        error: {
          code: "UNAUTHORIZED",
          message: "You need to log in first.",
        },
      },
      {
        status: 401,
        headers: {
          "Cache-Control": "no-store",
        },
      },
    );
  }

  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set(
    "redirect",
    `${request.nextUrl.pathname}${request.nextUrl.search}`,
  );

  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ["/app/:path*", "/api/pe-deal/:path*"],
};
