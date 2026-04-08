type ErrorBody = {
  error: {
    code: string;
    message: string;
  };
};

export class RouteError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export function assertSameOrigin(request: Request) {
  const origin = request.headers.get("origin");
  if (!origin) {
    throw new RouteError(
      403,
      "INVALID_ORIGIN",
      "Requests must come from the same origin.",
    );
  }

  const requestOrigin = new URL(request.url).origin;
  if (origin !== requestOrigin) {
    throw new RouteError(
      403,
      "INVALID_ORIGIN",
      "Cross-origin requests are not allowed.",
    );
  }
}

export async function parseJsonBody<T>(request: Request) {
  try {
    return (await request.json()) as T;
  } catch {
    throw new RouteError(400, "INVALID_JSON", "The request body must be valid JSON.");
  }
}

export function jsonNoStore(body: unknown, init?: ResponseInit) {
  const headers = new Headers(init?.headers);
  headers.set("Cache-Control", "no-store");
  return Response.json(body, {
    ...init,
    headers,
  });
}

export function toErrorResponse(error: unknown) {
  if (error instanceof RouteError) {
    return jsonNoStore(
      {
        error: {
          code: error.code,
          message: error.message,
        },
      } satisfies ErrorBody,
      { status: error.status },
    );
  }

  console.error(error);

  return jsonNoStore(
    {
      error: {
        code: "INTERNAL_ERROR",
        message: "Unexpected server error.",
      },
    } satisfies ErrorBody,
    { status: 500 },
  );
}

