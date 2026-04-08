import { randomBytes, scryptSync } from "node:crypto";

const password = process.argv[2];

if (!password) {
  console.error("Usage: npm run hash:password -- 'your-password'");
  process.exit(1);
}

const cost = 16384;
const blockSize = 8;
const parallelization = 1;
const salt = randomBytes(16);
const hash = scryptSync(password, salt, 64, {
  N: cost,
  r: blockSize,
  p: parallelization,
});

console.log(
  [
    "scrypt",
    cost,
    blockSize,
    parallelization,
    salt.toString("base64url"),
    hash.toString("base64url"),
  ].join("$"),
);

