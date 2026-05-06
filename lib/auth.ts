import { cookies } from "next/headers";
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

import { getAuthEnv } from "@/lib/env";
import { RouteError } from "@/lib/request";

const SESSION_COOKIE_NAME = "pe_deal_demo_session";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 12;

type SessionPayload = {
  sub: "shared-demo";
  iat: number;
  exp: number;
  nonce: string;
};

type ScryptHashParts = {
  cost: number;
  blockSize: number;
  parallelization: number;
  salt: Buffer;
  hash: Buffer;
};

export function isAuthenticatedRequest(request: Request) {
  const cookieHeader = request.headers.get("cookie");
  if (!cookieHeader) {
    return false;
  }

  const cookieValue = getCookieValue(cookieHeader, SESSION_COOKIE_NAME);
  if (!cookieValue) {
    return false;
  }

  return verifySessionCookieValue(cookieValue) !== null;
}

export function assertAuthenticatedRequest(request: Request) {
  if (!isAuthenticatedRequest(request)) {
    throw new RouteError(401, "UNAUTHORIZED", "You need to log in first.");
  }
}

export async function isCurrentSessionAuthenticated() {
  const cookieStore = await cookies();
  const cookieValue = cookieStore.get(SESSION_COOKIE_NAME)?.value;
  return cookieValue ? verifySessionCookieValue(cookieValue) !== null : false;
}

export async function createAuthenticatedSession() {
  const payload = signSessionPayload();
  const cookieStore = await cookies();

  cookieStore.set(SESSION_COOKIE_NAME, payload, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_MAX_AGE_SECONDS,
  });
}

export async function clearAuthenticatedSession() {
  const cookieStore = await cookies();

  cookieStore.set(SESSION_COOKIE_NAME, "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });
}

export function verifyAppGatePassword(password: string) {
  const { appGatePasswordHash } = getAuthEnv();
  const parsed = parseScryptHash(appGatePasswordHash);
  const derived = scryptSync(password, parsed.salt, parsed.hash.length, {
    N: parsed.cost,
    r: parsed.blockSize,
    p: parsed.parallelization,
  });

  return timingSafeEqual(parsed.hash, derived);
}

function signSessionPayload() {
  const { appGateSessionSecret } = getAuthEnv();
  const issuedAt = Math.floor(Date.now() / 1000);

  const payload: SessionPayload = {
    sub: "shared-demo",
    iat: issuedAt,
    exp: issuedAt + SESSION_MAX_AGE_SECONDS,
    nonce: randomBytes(12).toString("base64url"),
  };

  const payloadPart = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signaturePart = createHmac("sha256", appGateSessionSecret)
    .update(payloadPart)
    .digest("base64url");

  return `${payloadPart}.${signaturePart}`;
}

function verifySessionCookieValue(cookieValue: string) {
  const { appGateSessionSecret } = getAuthEnv();
  const [payloadPart, signaturePart] = cookieValue.split(".");

  if (!payloadPart || !signaturePart) {
    return null;
  }

  const expectedSignature = createHmac("sha256", appGateSessionSecret)
    .update(payloadPart)
    .digest("base64url");

  const expectedBuffer = Buffer.from(expectedSignature);
  const actualBuffer = Buffer.from(signaturePart);

  if (
    expectedBuffer.length !== actualBuffer.length ||
    !timingSafeEqual(expectedBuffer, actualBuffer)
  ) {
    return null;
  }

  try {
    const parsed = JSON.parse(
      Buffer.from(payloadPart, "base64url").toString("utf8"),
    ) as SessionPayload;

    if (parsed.sub !== "shared-demo" || parsed.exp <= Math.floor(Date.now() / 1000)) {
      return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

function parseScryptHash(input: string): ScryptHashParts {
  const parts = input.split("$");

  if (parts.length !== 6 || parts[0] !== "scrypt") {
    throw new Error(
      "APP_GATE_PASSWORD_HASH must use the `scrypt$N$r$p$salt$hash` format.",
    );
  }

  const cost = Number(parts[1]);
  const blockSize = Number(parts[2]);
  const parallelization = Number(parts[3]);
  const salt = Buffer.from(parts[4], "base64url");
  const hash = Buffer.from(parts[5], "base64url");

  if (
    Number.isNaN(cost) ||
    Number.isNaN(blockSize) ||
    Number.isNaN(parallelization) ||
    salt.length === 0 ||
    hash.length === 0
  ) {
    throw new Error("APP_GATE_PASSWORD_HASH is malformed.");
  }

  return {
    cost,
    blockSize,
    parallelization,
    salt,
    hash,
  };
}

function getCookieValue(cookieHeader: string, name: string) {
  const fragments = cookieHeader.split(";");

  for (const fragment of fragments) {
    const [rawKey, ...rest] = fragment.trim().split("=");
    if (rawKey === name) {
      return rest.join("=");
    }
  }

  return undefined;
}
