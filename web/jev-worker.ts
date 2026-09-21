import { createInterface } from 'node:readline';
import { evaluateJev } from './jev.js';
import { requireGatewayKey, safeFailure, safeErrorDetails } from './gateway-env.js';

// Long-lived SDK worker: benchmark latency excludes Node startup. JSONL stdin/stdout.
try { requireGatewayKey(); } catch { console.error('Gateway credentials are missing'); process.exit(2); }
for await (const line of createInterface({ input: process.stdin })) {
  const start = performance.now();
  try {
    const request = JSON.parse(line);
    const result = await evaluateJev(request.state, request.questions);
    console.log(JSON.stringify({ ok: true, latency_s: (performance.now() - start) / 1000, ...result }));
  } catch (error) {
    console.log(JSON.stringify({ ok: false, latency_s: (performance.now() - start) / 1000, error: safeFailure(error), ...safeErrorDetails(error) }));
  }
}
