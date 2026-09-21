import { loadEnvFile } from 'node:process';
import { fileURLToPath } from 'node:url';

// Server-side only. Never import this module into the browser application.
export function requireGatewayKey() {
  try {
    loadEnvFile(fileURLToPath(new URL('.env.local', import.meta.url)));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new Error('Cannot load local configuration');
  }
  if (!process.env.AI_GATEWAY_API_KEY?.trim()) {
    throw new Error('Missing AI_GATEWAY_API_KEY; enter it locally in web/.env.local');
  }
}

// SDK errors can contain request headers. Only allow a numeric status in logs.
export function safeFailure(error: unknown): string {
  const status = (error as { statusCode?: unknown } | null)?.statusCode;
  return typeof status === 'number' ? `Gateway request failed (HTTP ${status})` : 'Gateway request failed; check local credentials, model access and connectivity';
}

export function safeErrorDetails(error: unknown) {
  const status = (error as { statusCode?: unknown } | null)?.statusCode;
  const name = (error as { name?: unknown } | null)?.name;
  const category = typeof name === 'string' && (name.includes('Validation') || name.includes('InvalidResponse'))
    ? 'invalid_response' : 'request_failure';
  return { category, ...(typeof status === 'number' ? { status_code: status } : {}) };
}
